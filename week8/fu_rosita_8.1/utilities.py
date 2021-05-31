import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats as st
import scipy.special
import biocircuits
import warnings

import iqplot
import bokeh.io
import panel as pn


def style(p, autohide=False):
    p.title.text_font="Helvetica"
    p.title.text_font_size="16px"
    p.title.align="center"
    p.xaxis.axis_label_text_font="Helvetica"
    p.yaxis.axis_label_text_font="Helvetica"
    
    p.xaxis.axis_label_text_font_size="13px"
    p.yaxis.axis_label_text_font_size="13px"
    p.background_fill_alpha = 0
    if autohide: p.toolbar.autohide=True
    return p
    
    
gamma_N = 0.08
gamma_D = 0.08
gamma_R = 0.01
gamma_S = 0.1
k_t = 0.5
k_c = 5
k_RS = 1 / 1500
beta_N = 1
beta_R = 1.25e8
p = 2

D_plate_array = np.array(
    [0.063, 0.084, 0.11, 0.15, 0.20, 0.26, 0.35, 0.46, 0.62, 0.82, 1.1, 1.4]
)

beta_D = 0
N_trans = 0

Ro = 0
Do = 200
So = 0      # ignoring this for now
t = np.linspace(0, 100, 1000)


# ************************ COMPLEX ANALYSIS ************************ 

def derivs_A_supp(
    NDSRCT, t, 
    beta_N, beta_D, beta_R, 
    gamma_N, gamma_D, gamma_R, gamma_S, 
    k_c, k_t, k_RS,
    p,
    N_trans, D_trans
):
    N, D, S, R, C, T = NDSRCT
    dN_dt = beta_N - gamma_N*N - k_t*N*D_trans - k_c*N*D
    dD_dt = beta_D - gamma_D*D - k_t*N_trans*D - k_c*N*D
    dS_dt = k_t*N*D_trans - gamma_S*S
    dR_dt = beta_R*((k_RS*k_t*N*D_trans)/gamma_S)**p - gamma_R*R
    dC_dt = k_c*N*D
    dT_dt = k_c*N_trans*D + k_t*N*D_trans
    
    return np.array([dN_dt, dD_dt, dS_dt, dR_dt, dC_dt, dT_dt])
    
    
def plotter_complexes():
    D_plate_slider = pn.widgets.FloatSlider(name="D_plate", start=0.05, end=1.4, value=0.05, step=0.05, width=300)

    @pn.depends(D_plate_slider.param.value)
    def _plotter_complexes(D_plate):
        D_trans = D_plate
        No = beta_N / (gamma_N + k_c * Do + k_t * D_plate) 
        Co = k_c * No * Do
        To = k_t*(N_trans*Do + No*D_trans)
        args = (beta_N, beta_D, beta_R, 
            gamma_N, gamma_D, gamma_R, gamma_S, 
            k_c, k_t, k_RS, p, 
            N_trans, D_trans)
        NDSRCTo = np.array([No, Do, So, Ro, Co, To])
        t = np.linspace(0, 100, 1000)    
        
        _NDSRCT = scipy.integrate.odeint(derivs_A_supp, NDSRCTo, t, args)
        N_traj, D_traj, S_traj, R_traj, C_traj, T_traj = _NDSRCT.T    
        
        q = bokeh.plotting.figure(
            height=400, width=400, 
            title="(In)Activation Complexes", 
            x_axis_label="time (hrs)", y_axis_label="[complex]",
            y_range=(-2, 55)
        )
        q.line(t, C_traj, line_color="#bbbbbb", line_width=3, legend_label="cis inactivated complexes") # inactivation
        q.line(t, T_traj, line_color="#e3a201", line_width=3, legend_label="trans activated complexes")
        q.legend.location = 'top_left'
        q = style(q)
        
        return q
    lay_slider = pn.Row(D_plate_slider, align="center")
    
    return pn.Column(lay_slider, _plotter_complexes)
    
    
def plotter_shift():
    beta_D = 0
    D_plate = D_trans = 0.5

    beta_N_slider = pn.widgets.FloatSlider(name="βN", start=0.5, end=1.8, value=1, step=0.1, width=150)
    gamma_slider = pn.widgets.FloatSlider(name="γ", start=0.03, end=0.30, value=0.08, step=0.01, width=150)

    @pn.depends(beta_N_slider.param.value, gamma_slider.param.value)
    def plotter_shift(beta_N, gamma):
        gamma_N = gamma_D = gamma
        
        No = beta_N / (gamma_N + k_c * Do + k_t * D_plate) 
        Co = k_c * No * Do
        To = k_t*(N_trans*Do + No*D_trans)
        args = (beta_N, beta_D, beta_R, 
            gamma_N, gamma_D, gamma_R, gamma_S, 
            k_c, k_t, k_RS, p, 
            N_trans, D_trans)
        NDSRCTo = np.array([No, Do, So, Ro, Co, To])
        t = np.linspace(0, 100, 1000)    

        _NDSRCT = scipy.integrate.odeint(derivs_A_supp, NDSRCTo, t, args)
        N_traj, D_traj, S_traj, R_traj, C_traj, T_traj = _NDSRCT.T    

        q = bokeh.plotting.figure(
            height=450, width=400, 
            title="(In)Activation Complexes", 
            x_axis_label="time (hrs)", y_axis_label="[complex]",
            y_range=(-2, 55)
        )
        q.line(t, C_traj, line_color="#bbbbbb", line_width=3) # inactivation
        q.line(t, T_traj, line_color="#e3a201", line_width=3)

        legend = bokeh.models.Legend(items=[("cis inactivated complexes", [q.line(line_width=3, line_color="#bbbbbb")]),
                                            ("trans activated complexes", [q.line(line_width=3, line_color="#e3a201")])],
                                     location='center'
                                    )
        q.add_layout(legend, 'below')
        
        return style(q)

    lay_params = pn.Row(pn.Spacer(width=50), beta_N_slider, gamma_slider)
    return pn.Column(lay_params, plotter_shift, align="center")
    

# ************************ HOMOGENEOUS STEADY STATE ************************ 
def derivs_C(n1n2d1d2s1s2, t, alpha, beta_n, beta_d, gamma, kappa, kappa_s):
    n1, n2, d1, d2, s1, s2 = n1n2d1d2s1s2
    dn1_dt = beta_n - n1 - kappa*d1*n1 - d2*n1
    dn2_dt = beta_n - n2 - kappa*d2*n2 - d1*n2

    dd1_dt = beta_d/(1+s1**alpha) - d1 - kappa*d1*n1 - d1*n2
    dd2_dt = beta_d/(1+s2**alpha) - d2 - kappa*d2*n2 - d2*n1

    ds1_dt = kappa_s*d2*n1 - gamma*s1
    ds2_dt = kappa_s*d1*n2 - gamma*s2

    return np.array([dn1_dt, dn2_dt, dd1_dt, dd2_dt, ds1_dt, ds2_dt])
    
# intersection code lifted from pip package `intersection`
def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4

def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj

def intersection(x1, y1, x2, y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
usage:
x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)
    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]


def plotter_homogeneous(color_n1='#5779a3', color_n2='#84b5b2', color_d1='#d1605e', color_d2='#e59244', color_s1='#808bbe', color_s2='#b09dc9'):
    alpha = 2
    no, do = 5, 100
    n1o, n2o = no, no
    d1o, d2o = do, do
    s1o, s2o = 0, 0
    n1n2d1d2s1s2_o = np.array([n1o, n2o, d1o, d2o, s1o, s2o])
    
    beta_n_slider = pn.widgets.FloatSlider(name='βn', start=0.1, end=10, step=0.1, value=8.60, width=95)
    beta_d_slider = pn.widgets.IntSlider(name='βd', start=1, end=30, step=1, value=18, width=95)
    gamma_slider = pn.widgets.FloatSlider(name='γ', start=1, end=20, step=1, value=5, width=95)
    kappa_slider = pn.widgets.FloatSlider(name='κ', start=1, end=10, step=0.5, value=10, width=95)
    kappa_s_slider = pn.widgets.FloatSlider(name='κS', start=1, end=100, step=2, value=20, width=95)

    show_S_toggle = pn.widgets.Toggle(name="SHOW S", value=False, width=200)
    t_max_slider = pn.widgets.IntSlider(name="t-max", start=15, end=60, step=5, value=25, width=200)
    y_max_slider = pn.widgets.IntSlider(name="y-max", start=6, end=30, step=3, value=6, width=200)

    @pn.depends(beta_n_slider.param.value, beta_d_slider.param.value, 
                gamma_slider.param.value, kappa_slider.param.value, kappa_s_slider.param.value,
                y_max_slider.param.value, t_max_slider.param.value, show_S_toggle.param.value,
               )
    def _plotter_find_params(beta_n, beta_d, gamma, kappa, kappa_s, y_max, t_max, show_S):
        do_space = np.logspace(-5, 5, 200)
        no_space = np.logspace(-5, 5, 200)

        # ..... NULLCLINES .... 
        no = beta_n / (1+do_space*(1+kappa))
        do = beta_d / (1+(kappa_s/(gamma*(1+kappa))*(beta_n - no_space))**alpha) / (1+no_space*(1+kappa))
        
        # finding fixed point intersection
        x, y = intersection(no_space, do, no, do_space)
        
        p = bokeh.plotting.figure(
                height=325, width=375, 
                x_axis_type='log', y_axis_type='log',
                x_range=(1e-3, 1e3), y_range=(1e-3, 1e3),
                x_axis_label="n", y_axis_label="d",
                title='Nullclines',
            )
        t = np.linspace(0, t_max, 300)
        p.line(no_space, do, line_width=3, line_color='#3d2314')
        p.line(no, do_space, line_width=3, line_color='#3d2314')
        p.circle(x, y, size=13, line_color='#3d2314', fill_color='white', line_width=4)

        # .... TRAJECTORIES .... 
        args = (alpha, beta_n, beta_d, gamma, kappa, kappa_s)
        _n1n2d1d2s1s2 = scipy.integrate.odeint(derivs_C, n1n2d1d2s1s2_o, t, args)
        n1_traj, n2_traj, d1_traj, d2_traj, s1_traj, s2_traj = _n1n2d1d2s1s2.T
        q = bokeh.plotting.figure(
            height=325, width=475,
            x_axis_label='time', y_axis_label='[ ]',
            title='Trajectories',
            y_range=(-0.2, y_max+0.2)
        )
        grey = "#bbbbbb"

        if show_S:
            q.line((t[0], t[-1]), (x, x), line_width=2, line_color=grey, line_dash='dotdash')
            q.line((t[0], t[-1]), (y, y), line_width=2, line_color=grey, line_dash='dotdash')
            q.line(t, n1_traj, line_width=2, line_color=grey)
            q.line(t, n2_traj, line_width=2, line_color=grey)
            q.line(t, d1_traj, line_width=2, line_color=grey)
            q.line(t, d2_traj, line_width=2, line_color=grey)
            q.line(t, s2_traj, line_width=4, line_color=color_s2)
            q.line(t, s1_traj, line_width=4, line_color=color_s1)
            legend = bokeh.models.Legend(items=[
                ("s1", [q.line(line_width=3, line_color=color_s1)]),
                ("s2", [q.line(line_width=3, line_color=color_s2)]),
            ], location='center')
        else: 
            q.line((t[0], t[-1]), (x, x), line_width=4, line_color=grey, line_dash='dotdash')
            q.line((t[0], t[-1]), (y, y), line_width=4, line_color=grey, line_dash='dotdash')

            q.line(t, n1_traj, line_width=3, line_color=color_n1)
            q.line(t, n2_traj, line_width=3, line_color=color_n2)
            q.line(t, d1_traj, line_width=3, line_color=color_d1)
            q.line(t, d2_traj, line_width=3, line_color=color_d2)
            legend = bokeh.models.Legend(items=[
                ("n1", [q.line(line_width=3, line_color=color_n1)]),
                ("n2", [q.line(line_width=3, line_color=color_n2)]),
                ("d1", [q.line(line_width=3, line_color=color_d1)]),
                ("d2", [q.line(line_width=3, line_color=color_d2)]),
                ("fp", [q.line(line_width=3, line_color=grey, line_dash="dotdash")])
            ], location='center')
            
        q.add_layout(legend, 'right')
        
        return pn.Row(style(p), style(q))

    lay_params = pn.Column(
        pn.Spacer(height=10),
        beta_n_slider, beta_d_slider, gamma_slider, 
        kappa_slider, kappa_s_slider
    )
    lay_traj = pn.Column(
        _plotter_find_params, 
        pn.Row(pn.Spacer(width=450), show_S_toggle),
        pn.Row(pn.Spacer(width=450), y_max_slider),
        pn.Row(pn.Spacer(width=450), t_max_slider),
    )
    return pn.Row(lay_params, lay_traj)


# ************************ LATTICE MODELING ************************ 
def plotter_lattice_trajectories(t, trajs, color_N, color_D):
    p = bokeh.plotting.figure(
        height=400, width=575,
        title="Lattice Trajectories",
        x_axis_label="time",
        y_axis_label="[ ]"
    )
    lw = 2

    for traj in trajs[:60]: 
        p.line(t, traj, line_width=lw, line_color=color_N, line_alpha=0.5)
    for traj in trajs[60:120]: 
        p.line(t, traj, line_width=lw, line_color=color_D, line_alpha=0.5)
    legend = bokeh.models.Legend(items=[
        ("N", [p.line(line_color=color_N, line_width=lw)]),
        ("D", [p.line(line_color=color_D, line_width=lw)]), 
    ], location='center')
    p.add_layout(legend, 'right')
    return style(p)
    
    
def hex_to_rgb(palette):
    if type(palette)==str: return tuple([int(palette.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)])
    return [tuple(int(h.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) for h in palette]

def rgb_to_hex(palette):
    if type(palette[0])==int: return "#%02x%02x%02x" % palette
    return ["#%02x%02x%02x" % (r,g,b) for (r,g,b) in palette]

def get_D_colors(trajs, color_D):
    color_D_rgb = hex_to_rgb(color_D)
    N_trajs, D_trajs, S_trajs = trajs[:60], trajs[60:120], trajs[120:]
    D_max = max(D_trajs.flatten())

    color_D_rgb = hex_to_rgb(color_D)

    D_colors, D_alphas = [], []
    for D_traj in D_trajs:
        _D_colors, _D_alphas = [], []
        for D in D_traj:
            f = D / D_max
            _color = f
            _color_rgb = [int(_) for _ in f * np.array(color_D_rgb)]
            _color = rgb_to_hex(tuple(_color_rgb))
            _D_colors.append(_color)
            _D_alphas.append(f)
        D_colors.append(_D_colors)
        D_alphas.append(_D_alphas)
        
    return D_colors, D_alphas

def rearrange(D_time):
    rearranged = []
    for i in range(10):
        for j in range(i, i+60, 10):
            rearranged.append(D_time[j])
    return rearranged
    

def plotter_lattice_hexagons(t, trajs, color_D):
    D_colors, D_alphas = get_D_colors(trajs, color_D)
    time_slider = pn.widgets.IntSlider(name="time", start=0, end=len(t)-1, value=0)
    
    r = [_ for _ in range(6)]*10
    _q = [0, -1, -1, -2, -2, -3]
    __q = [[_ + __ for _ in _q] for __ in range(10)]
    q = [_ for __ in __q for _ in __]
    
    @pn.depends(time_slider.param.value)
    def _plotter_lattice_hexagons(time):
        D_time_colors = [D_traj[time] for D_traj in D_colors]
        D_time_alphas = [D_traj[time] for D_traj in D_alphas]
        
        colors = rearrange(D_time_colors)
        alphas = rearrange(D_time_alphas)

        source = bokeh.models.ColumnDataSource(dict(r=r, q=q, color=colors, alpha=alphas))
        
        viz1 = bokeh.models.Plot(plot_width=600, plot_height=300, toolbar_location=None)
        viz2 = bokeh.models.Plot(plot_width=600, plot_height=300, toolbar_location=None)
        glyph1 = bokeh.models.HexTile(
            q='q',r='r', 
            line_color="white", 
            fill_color="color", 
        )
        glyph2 = bokeh.models.HexTile(
            q='q',r='r', 
            fill_color=color_D,
            fill_alpha="alpha",
            line_color="black", 
        )
        viz1.add_glyph(source, glyph1)
        viz2.add_glyph(source, glyph2)
        return pn.Column(viz2, viz1)

    return pn.Column(time_slider, _plotter_lattice_hexagons)
    
    
