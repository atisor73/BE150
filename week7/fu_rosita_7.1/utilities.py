import json
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats as st
import scipy.special
import numba
from tqdm import tqdm
import biocircuits
import warnings

import iqplot
import bokeh.io
import panel as pn

warnings.filterwarnings('ignore')

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
    
    
    
    
    
    

def dc_dt(c, m, alpha_c, beta_o, gamma):
    term1 = beta_o * m**2*c / (1 + m**2)
    term2 = - gamma * c
    term3 = - alpha_c * m * c
    return term1 + term2 + term3

def dm_dt(c, m, alpha_o, Io):
    term1 = Io + c
    term2 = - alpha_o * m**2 * c / (1 + m**2)
    term3 = - m
    return term1 + term2 + term3

def ode_rhs(x, t, alpha_o, alpha_c, beta_o, gamma, Io):
    c, m = x
    dm_dt = Io + c - alpha_o * c * m**2/(1+m**2) - m
    dc_dt = beta_o * m**2*c/(1+m**2) - gamma*c - alpha_c * m * c
    
    return np.array([dc_dt, dm_dt])


def lin_stab_matrix(c, m, alpha_o, alpha_c, beta_o, gamma, Io):
    e00 = beta_o * m**2 / (1+m**2)  - gamma - alpha_c * m     # df(c)/dc
    e11 = - alpha_o * c * 2*m/((1+m**2)**2) - 1               # df(m)/dm
    
    e01 = beta_o*2*c*m/((1+m**2)**2) - alpha_c * c            # df(c)/dm
    e10 = 1 - alpha_o * m**2/(1+m**2)                         # df(m)/dc
    
    A = np.array([[e00, e01], [e10, e11]])
    return A

def lin_stab(c, m, alpha_o, alpha_c, beta_o, gamma, Io):
    A = lin_stab_matrix(c, m, alpha_o, alpha_c, beta_o, gamma, Io)
    
    return np.linalg.eig(A)
    

def plot_separatrix(
    c,
    m,
    c_range,
    m_range,
    p,
    args,
    t_max=50,
    eps=1e-6,
    color="#d95f02",
    line_width=2,
    line_dash="solid",
    log=False,
    return_separatrix=False,
):
    # Negative time function to integrate to compute separatrix
    def rhs(x, t):
        c, m = x

        # Stop integrating if we get the edge of where we want to integrate
        if c_range[0] < c < c_range[1] and m_range[0] < m < m_range[1]:
            return -ode_rhs(x, t, *args)
        else:
            return np.array([0, 0])

    evals, evecs = lin_stab(c, m, *args)
    evec = evecs[:, evals < 0].flatten()   # Get eigenvector corresponding to attraction
    
    t = np.linspace(0, t_max, 1000)

    # Build upper right branch of separatrix
    x0 = np.array([c, m]) + eps * evec
    x_upper = scipy.integrate.odeint(rhs, x0, t)

    # Build lower left branch of separatrix
    x0 = np.array([c, m]) - eps * evec
    x_lower = scipy.integrate.odeint(rhs, x0, t)

    # Concatenate, reversing lower so points are sequential
    sep_c = np.concatenate((x_lower[::-1, 0], x_upper[:, 0]))
    sep_m = np.concatenate((x_lower[::-1, 1], x_upper[:, 1]))

    if return_separatrix: return sep_c, sep_m
    
    # Plot
    if log:
        p.line(np.log(sep_c), np.log(sep_m), color=color, line_width=line_width, line_dash=line_dash)
    else: 
        p.line(sep_c, sep_m, color=color, line_width=line_width, line_dash=line_dash)
        
    return p
    
### *********************** PART (E), (F) FUNCTION ************************
def part_ef_plotter(sep_c, sep_m):
    beta_o = 9.78
    gamma = 2.14
    Io = 24.67
    alpha_c = 1.06
    alpha_o = 11.21
    
    color_separatrix = "red"
    color_dc_dt = "#9fc0c1"
    color_dm_dt = "#eba8b5"
    color_phase = "#cacdca"
    color_fixed = "#65042d"
    
    c0_st = 0
    m0_st = Io
    _roots = np.roots([alpha_c, gamma-beta_o, alpha_c, gamma])
    m1_st, m2_st = [_root for _root in _roots if _root >= 0]
    find_c_cubic = lambda m : (m**3 - Io*m**2 + m - Io) / (1 + (m**2)*(1-alpha_o))
    c1_st, c2_st = find_c_cubic(m1_st), find_c_cubic(m2_st)
    
    
    c_range=(-1.5, 13)
    m_range=(-0.28, 28)
    params_c = (alpha_c, beta_o, gamma)
    params_m = (alpha_o, Io)

    p = bokeh.plotting.figure(
        height=500,
        width=650,
        title="Nullclines on the 𝑐−𝑚 plane",
        x_axis_label="c",
        y_axis_label="m",
    )

    # .... PHASE PORTRAIT ....
    p = biocircuits.phase_portrait(
        dc_dt, 
        dm_dt, 
        c_range, 
        m_range, 
        params_c, 
        params_m, 
        x_axis_label="c", 
        y_axis_label="m", 
        color=color_phase,
        title="Nullclines",
        p=p,
        density=2
    )

    # .... NULLCLINES ....
    lw = 4.5
    c_space = np.linspace(c_range[0], c_range[1], 1000)
    m_space = np.linspace(m_range[0], m_range[1], 1000)

    c_zero = 0 * m_space
    c_cubic = find_c_cubic(m_space)

    p.line(c_zero, m_space, line_width=lw, line_color=color_dc_dt)
    p.line(c_space, m1_st, line_width=lw, line_color=color_dc_dt)
    p.line(c_space, m2_st, line_width=lw, line_color=color_dc_dt)
    p.line(c_cubic, m_space, line_width=lw, line_color=color_dm_dt)

    # .... SEPARATRIX .... 
    args = (alpha_o, alpha_c, beta_o, gamma, Io)
    fps = np.array([[c0_st, m0_st], [c1_st, m1_st], [c2_st, m2_st]])
    c_unstable, m_unstable = fps[1]
    p.line(sep_c, sep_m, color=color_separatrix, line_width=7, line_dash="dotdash")

    # .... FIXED POINTS ....
    # unstable unfilled hole
    p.circle(
        c1_st, m1_st, 
        size=12, 
        color="white", 
        line_color="black", 
        line_width=4
    ) 
    # pizazz wedges
    spin = 0
    start_angles = np.linspace(0, 2*np.pi, 16)
    end_angles = start_angles + 0.1
    for start_angle, end_angle in zip(start_angles, end_angles):
        out_radius = 0.1
        p.annular_wedge(
            x=c1_st, 
            y=m1_st, 
            fill_color="black",
            line_color="black",
            inner_radius=0.4, 
            outer_radius=0.4+out_radius, 
            start_angle=start_angle+spin, 
            end_angle=end_angle+spin, 
        )
        
    # stable filled hole
    p.circle(c0_st, m0_st, size=16, color="black", line_color="white", line_width=1.8)
    p.circle(c2_st, m2_st, size=16, color="black", line_color="white", line_width=1.8)

    # .... LEGEND .... 
    legend = bokeh.models.Legend(items=[
        ("stable fp", [p.circle(color="black", size=15)]), 
        ("unstable fp", [p.circle(color="white", size=15, line_color="black", line_width=2)]), 
        ("dc/dt nullclines", [p.line(color=color_dc_dt, line_width=lw)]),
        ("dm/dt nullcline", [p.line(color=color_dm_dt, line_width=lw)]),
        ("separatrix", [p.line(color=color_separatrix, line_width=5,  line_dash="dotdash")]), 
    ], location="center")
    p.add_layout(legend, 'right')

    return style(p)


### ************************* PART (G) FUNCTION *************************
def part_g_plotter(sep_c, sep_m):
    color_M = "#db98a5"
    color_C = "#9ab5b6"

    beta_o = 9.78
    gamma = 2.14
    Io = 24.67
    alpha_c = 1.06
    alpha_o = 11.21
    
    args = (alpha_o, alpha_c, beta_o, gamma, Io)
    
    LOG_toggle = pn.widgets.Toggle(name="LOG")
    @pn.depends(LOG_toggle.param.value)
    def _part_g_plotter(LOG):
        y_axis_type = "log" if LOG else "linear"
        y_axis_label = "log [ ]" if LOG else "[ ]"
        
        q = bokeh.plotting.figure(
            height=360, 
            width=600, 
            title="Below the Separatrix",
            x_axis_label="dimensionless time",
            y_axis_label=y_axis_label,
            y_axis_type=y_axis_type
        )

        t_max = 3
        t = np.linspace(0, t_max, 1000)

        offset = 2
        Cs_above, Ms_above = sep_c[1020:1080], sep_m[1020:1080] - offset

        for C, M in zip(Cs_above, Ms_above):
            CMo = np.array([C, M])
            _CM = scipy.integrate.odeint(ode_rhs, CMo, t, args=args)
            C, M = _CM.T

            q.line(t, M, line_color=color_M, line_width=2.5, line_alpha=0.6)
            q.line(t, C, line_color=color_C, line_width=2.5, line_alpha=0.6)
        q.line(x=(0, t_max), y=(0, 0), line_color="black", line_width=3, line_dash="dotdash")
        
        legend = bokeh.models.Legend(items=[
            ("m above sep", [q.line(line_color=color_M, line_width=3)]), 
            ("c above sep", [q.line(line_color=color_C, line_width=3)]), 
            ("[ ] = 0", [q.line(line_color="black", line_width=3, line_dash="dotdash")])
        ], location="center")
        q.add_layout(legend, 'right')

        return style(q)
        
    dashboard = pn.Column(_part_g_plotter, 
        pn.Row(pn.Spacer(width=70), 
        LOG_toggle)
    )
    return dashboard

### ************************* PART (H) FUNCTIONS *************************
def draw_fp(x, y, p, LOG, fp_type="stable", in_radius=0.8, out_radius=0.3):
    if fp_type == "stable":
        p.circle(x, y, size=16, color="black", line_color='white', line_width=0.8)
        
    if fp_type == "unstable":
        p.circle(x, y, size=12, color="white", line_color="black", line_width=4)
        
        # pizazz wedges
        spin = 0
        start_angles = np.linspace(0, 2*np.pi, 12)
        end_angles = start_angles + 0.1

        in_radius = in_radius if LOG else 0.5
        out_radius = out_radius if LOG else 0.2

        for start_angle, end_angle in zip(start_angles, end_angles):
            p.annular_wedge(
                x=x, 
                y=y, 
                fill_color="black",
                line_color="black",
                inner_radius=in_radius, 
                outer_radius=in_radius+out_radius, 
                start_angle=start_angle+spin, 
                end_angle=end_angle+spin, 
            )

    return p

def find_c_cubic(m, alpha_o, Io):
    return (m**3 - Io*m**2 + m - Io) / (1 + (m**2)*(1-alpha_o))
    


def param_plotter():
    color_dc_dt = "#9fc0c1"
    color_dm_dt = "#ecbac5"
    color_fixed = "#65042d"

    alpha_c_slider = pn.widgets.FloatSlider(name="αc", start=0.01, end=5, step=0.01, value=1.06, width=175)
    alpha_o_slider = pn.widgets.FloatSlider(name="αο", start=0.01, end=20, step=0.1, value=11.21, width=175)
    beta_o_slider = pn.widgets.FloatSlider(name="βο", start=0.01, end=20, step=0.1, value=9.78, width=175)
    gamma_slider = pn.widgets.FloatSlider(name="γ", start=0.01, end=5, step=0.1, value=2.14, width=175)
    Io_slider = pn.widgets.FloatSlider(name="Io", start=15, end=30, step=1, value=24.67, width=175)
    LOG_toggle = pn.widgets.Toggle(name="LOG", width=300)

    @pn.depends(LOG_toggle.param.value, alpha_c_slider.param.value, alpha_o_slider.param.value, 
                beta_o_slider.param.value, gamma_slider.param.value, Io_slider.param.value)
    def _param_plotter(LOG, alpha_c, alpha_o, beta_o, gamma, Io):
        params_c = (alpha_c, beta_o, gamma)
        params_m = (alpha_o, Io)
        args = (alpha_o, alpha_c, beta_o, gamma, Io)
        
        # finding fixed points
        c0_st = 0
        m0_st = Io
        _roots = np.roots([alpha_c, gamma-beta_o, alpha_c, gamma])
        ms_st = [_root for _root in _roots if _root >= 0]
        cs_st = [find_c_cubic(m, alpha_o, Io) for m in ms_st]
        
        stabilities = []
        # determine stability of fixed points
        for m, c in zip(ms_st, cs_st):
            evals, evecs = lin_stab(c, m, *args)
            # not 100% how stability analysis goes but we'll go with it for now...
            if np.all(evals < 0): stabilities.append(True)
            else: stabilities.append(False) 
        
        # initialize plotting variables
        x_axis_type = "log" if LOG else "linear"  
        y_axis_type = "log" if LOG else "linear"  
        x_axis_label = "log(c)" if LOG else "c"
        y_axis_label = "log(m)" if LOG else "m"
        x_range = (1e-1, 1e3) if LOG else (-1, 16)
        y_range = (1e-2, 1e3) if LOG else (-1, 29)
        c_range = (-5, 4) if LOG else (-1.5, 100)
        m_range = (-7, 7) if LOG else (-0.28, 100)
        
        # refining plot ranges
        _cmax_fp = max(cs_st)
        _mmax_fp = max(ms_st)
        if _cmax_fp > x_range[1]:
            x_range = (x_range[0], _cmax_fp+50) if LOG else (x_range[0], _cmax_fp+3)
        if _mmax_fp > y_range[1]:
            y_range = (y_range[0], _mmax_fp+50) if LOG else (y_range[0], _mmax_fp+3)
        p = bokeh.plotting.figure(
            height=400,
            width=560,
            title="Nullclines on the 𝑐−𝑚 plane",
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            x_range=x_range,
            y_range=y_range,
            x_axis_type=x_axis_type,
            y_axis_type=y_axis_type,
        )

        # .... NULLCLINES ....
        lw = 3
        if LOG: 
            c_space = np.logspace(c_range[0], c_range[1], 1000)
            m_space = np.logspace(m_range[0], m_range[1], 1000)
        else: 
            c_space = np.linspace(c_range[0], c_range[1], 1000)
            m_space = np.linspace(m_range[0], m_range[1], 1000)

        c_zero = 0 * m_space
        c_cubic = find_c_cubic(m_space, alpha_o, Io)

        p.line(c_cubic, m_space, line_width=lw, line_color=color_dm_dt)
        p.line(c_zero, m_space, line_width=lw, line_color=color_dc_dt)
        for m in ms_st: 
            try: 
                p.line(c_space, m, line_width=lw, line_color=color_dc_dt)
            except: 
                pass

        # .... FIXED POINTS ....
        draw_fp(c0_st, m0_st, p, LOG, fp_type="stable")
        for c, m, stab in zip(cs_st, ms_st, stabilities):
            fp_type = "stable" if stab else "unstable"
            try: 
                p = draw_fp(c, m, p, LOG, fp_type=fp_type)
            except: 
                pass

            
        # .... LEGEND .... 
        legend = bokeh.models.Legend(items=[
            ("stable fp", [p.circle(color="black", size=15)]), 
            ("unstable fp", [p.circle(color="white", size=15, line_color="black", line_width=2)]), 
            ("dc/dt nullclines", [p.line(color=color_dc_dt, line_width=lw)]),
            ("dm/dt nullcline", [p.line(color=color_dm_dt, line_width=lw)]),
        ], location="center")
        p.add_layout(legend, 'right')
        
        return style(p, autohide=True)

    lay_params = pn.Row(
        pn.Spacer(width=10),
        pn.Column(alpha_c_slider, beta_o_slider, gamma_slider), 
        pn.Column(pn.Spacer(height=10), alpha_o_slider, Io_slider)
    )
    lay_logtoggle = pn.Row(pn.Spacer(width=40), LOG_toggle)
    dashboard = pn.Column(
        lay_params,
        _param_plotter, 
        lay_logtoggle
    )
    
    return dashboard
    
    
def classifier(alpha_o, alpha_c, beta_o, gamma, Io):
    """
    Returns 0: positive real m, positive c NOT found
    Returns 1: positive real m, positive c, unstable fp
    Returns 2: positive real m, positive c, stable fp
    """
    ms_st = []
    _roots = np.roots([alpha_c, gamma-beta_o, alpha_c, gamma])
    for _root in _roots:
        if (np.imag(_root) == 0.0) & (np.real(_root) > 0.0):
            ms_st.append(_root)
    cs_st = [find_c_cubic(m, alpha_o, Io) for m in ms_st]
    
    if len(ms_st) == 0: 
        return 0
    
    args = (alpha_o, alpha_c, beta_o, gamma, Io)
    for m, c in zip(ms_st, cs_st):
        evals, evecs = lin_stab(c, m, *args)
        if (np.all(evals < 0)) and (c > 0):
            return 2
        
    return 1

def stab_abig_plotter(d_abg, d_aig):
    ng = 20
    gamma_range = np.logspace(-1, 2, ng)
    color_dict = {0: '#eba8b5', 1:'#9fc0c1', 2:'#65042d'}

    df_abg = pd.DataFrame(d_abg)
    df_aig = pd.DataFrame(d_aig)
    df_aig = df_aig[::2]
    
    index_gamma_slider = pn.widgets.IntSlider(name="index γ", 
        start=0, end=ng-1, width=250, value_throttled=0)
    @pn.depends(index_gamma_slider.param.value)
    def _stab_abig_plotter(index_gamma):
        gamma = gamma_range[index_gamma]
        
        p = bokeh.plotting.figure(
            x_axis_label="αc", y_axis_label="βo", 
            x_axis_type="log", y_axis_type="log",
            height=375, width=375,
            title=f"αc-βo plane, γ: {np.round(gamma, 1)}",
        )
        q = bokeh.plotting.figure(
            x_axis_label="αo", y_axis_label="𝐼o", 
            x_axis_type="log", y_axis_type="log",
            height=375, width=500,
            title=f"𝐼o-αo plane, γ: {np.round(gamma, 1)}",
        )
        sub_df_abg = df_abg.loc[df_abg['gamma']==gamma]
        sub_df_aig = df_aig.loc[df_aig['gamma']==gamma]
        
        p.circle(source=sub_df_abg, x='alpha',y='beta', color='color', size=4)
        q.circle(source=sub_df_aig, x='alpha_o',y='Io', color='color', size=4)
        
        
        legend = bokeh.models.Legend(items=[
            ("no fp", [q.circle(color=color_dict[0])]), 
            ("unstable fp", [q.circle(color=color_dict[1])]), 
            ("stable fp", [q.circle(color=color_dict[2])])
        ], location="center")
        q.add_layout(legend, 'right')
        
        p = style(p, autohide=True)
        q = style(q, autohide=True)
        
        return pn.Row(p, q)

    plots = pn.Column(
        _stab_abig_plotter, 
        pn.Row(pn.Spacer(width=200), 
        index_gamma_slider)
    )
    dashboard = pn.Row(plots, align="center")
    return dashboard
    
    
