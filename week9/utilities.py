import math
import numpy as np
import pandas as pd
import scipy.integrate
import biocircuits
import tqdm

import iqplot
import bokeh.io
import panel as pn

def style(p, autohide=False, toolbar_location="right"):
    p.title.text_font="Helvetica"
    p.title.text_font_size="16px"
    p.title.align="center"
    p.xaxis.axis_label_text_font="Helvetica"
    p.yaxis.axis_label_text_font="Helvetica"
    
    p.xaxis.axis_label_text_font_size="13px"
    p.yaxis.axis_label_text_font_size="13px"
    p.background_fill_alpha = 0
    
    p.toolbar_location = toolbar_location
    if autohide: p.toolbar.autohide=True
    return p
    
    
    
def interpolate(t, X, new_t):
    spl = scipy.interpolate.splrep(t, X) # B-spline
    X_delay = scipy.interpolate.splev(new_t, spl)
    return X_delay

def derivs(
    X, t, 
    N_delay, Pa_delay_M, Pa_delay_W, 
    β_p, β_sp, β_m, β_mi, β_w, β_s,
    γ_pi, γ_m, γ_w, γ_s, 
    δ_mpi, δ_mpa, δ_sm, δ_ws,
    k, k_w, k_s,
    n_n, n_s, n_w
):
    Pi, Pa, M, W, S = X          # unpacking vector
    
    dPi_dt = ( β_p - γ_pi*Pi 
               - β_sp*Pi * (S/k_s)**n_s/(1+(S/k_s)**n_s) 
               - δ_mpi*M*Pi/(1+(N_delay/k)**n_n) )
    dPa_dt = (   β_sp*Pi * (S/k_s)**n_s/(1+(S/k_s)**n_s) 
               - δ_mpa*M*Pa/(1+(N_delay/k)**n_n) )
    dM_dt = β_mi - γ_m*M + β_m*Pa_delay_M - δ_sm*S*M
    dW_dt =      - γ_w*W + β_w*Pa_delay_W
    dS_dt = β_s - γ_s*S - δ_ws*S*(W/k_w)**n_w/(1+(W/k_w)**n_w)
    
    return np.array([dPi_dt, dPa_dt, dM_dt, dW_dt, dS_dt])


def find_index_nearest(arr,value):
    i = np.searchsorted(arr, value, side="left")
    if i > 0 and (i==len(arr) or math.fabs(value-arr[i-1]) < math.fabs(value-arr[i])):
        return i-1
    else:
        return i

floor = lambda Y: Y if Y > 0 else 0 

# ************************************* NO NUTLIN *************************************
def trajectory(
    N_tune,
    Pi_o, Pa_o, M_o, W_o, S_o,
    to, tf, timestep, 
    τ_N, τ_Μ, τ_W, 
    args
):
    t_range = np.linspace(to, tf, timestep)            # setting up time array
    if len(N_tune) != len(t_range):                    # checking dimension of tuned N
        print("`N_tune` needs to match dimensions of time array set up by to, tf, timestep")
        return
    
    Pi, Pa, M, W, S = Pi_o, Pa_o, M_o, W_o, S_o        # initializing.... 
    Pi_traj, Pa_traj, M_traj, W_traj, S_traj = [Pi_o], [Pa_o], [M_o], [W_o], [S_o]

    for i in range(len(t_range)-1):                    # loop integration
        t = [t_range[i], t_range[i+1]]
        Xo = np.array([Pi, Pa, M, W, S])
        
#         N_delay = N_tune[i] if t[0] < 0.4 else interpolate(t_range[:i], N_tune[:i], t[0]-0.4)
        if t[0] < 0.4: 
            N_delay = N_tune[i]
        else: 
            _index = find_index_nearest(t_range, t[0]-0.4)
            N_delay = N_tune[_index]
        Pa_delay_M = Pa_o if t[0] < 0.7 else interpolate(t_range[:i], Pa_traj[:i], t[0]-0.7)
        Pa_delay_W = Pa_o if t[0] < 1.25 else interpolate(t_range[:i], Pa_traj[:i], t[0]-1.25)
    
        args_all = (N_delay, Pa_delay_M, Pa_delay_W, *args)

        _Pi, _Pa, _M, _W, _S = scipy.integrate.odeint(derivs, Xo, t, args=args_all).T
        Pi, Pa, M, W, S = _Pi[-1], _Pa[-1], _M[-1], _W[-1], _S[-1]
        Pi, Pa = Pi if Pi > 0 else 0, Pa if Pa > 0 else 0
        M, W, S = M if M > 0 else 0, W if W > 0 else 0, S if S > 0 else 0
        
        Pi_traj.append(Pi)
        Pa_traj.append(Pa)        
        M_traj.append(M)
        W_traj.append(W)
        S_traj.append(S)
    Pi_traj, Pa_traj = np.array(Pi_traj), np.array(Pa_traj)
    M_traj, W_traj, S_traj = np.array(M_traj), np.array(W_traj), np.array(S_traj)
    
    return t_range, Pi_traj, Pa_traj, M_traj, W_traj, S_traj

def A_response_times_plotter(SIM_time, S_traj, Pa_traj, M_traj, W_traj):
    p = bokeh.plotting.figure(
        height=300, width=700, title="p53, Mdm2, Wip1 response to DNA damage signal",
        x_axis_label="time (hrs)", y_axis_label="[ ] (mM)"
    )
    p.line(SIM_time, S_traj, line_width=3, color='#67001E')
    p.line(SIM_time, Pa_traj, line_width=3, color='#B2172B')
    p.line(SIM_time, M_traj, line_width=3, color='#D65E4C') 
    p.line(SIM_time, W_traj, line_width=3, color='#F4A582') 
    legend = bokeh.models.Legend(items=[
        ("damage", [p.line(line_width=3, color="#67001E")]),
        ("active p53", [p.line(line_width=3, color="#B2172B")]),
        ("Mdm2", [p.line(line_width=3, color="#D65E4C")]),
        ("Wip1", [p.line(line_width=3, color="#F4A582")]),
    ], location='center')
    p.add_layout(legend, 'right')
    p.legend.click_policy="hide"
    p = style(p, autohide=True, toolbar_location='left')
    return p
    
    
def A_total_p53_plotter(SIM_time, Pi_traj, Pa_traj, SIM_total_p53_conc):
    p = bokeh.plotting.figure(
        height=300, width=700, title="👀 total p53",
        x_axis_label="time (hrs)", y_axis_label="[ ] (mM)"
    )
    p.line(SIM_time, Pi_traj, line_width=3, color='#D4B3CF')
    p.line(SIM_time, Pa_traj, line_width=3, color='#B09DC9')
    p.line(SIM_time, SIM_total_p53_conc, line_width=3, color='black') 
    legend = bokeh.models.Legend(items=[
        ("inactive p53", [p.line(line_width=3, color="#D4B3CF")]),
        ("active p53", [p.line(line_width=3, color="#B09DC9")]),
        ("total p53", [p.line(line_width=3, color="black")]),
    ], location='center')
    p.add_layout(legend, 'right')
    p.legend.click_policy="hide"
    p = style(p, autohide=True, toolbar_location='left')
    
    return p
    
    
def A_empirical_plotter(
    SIM_time, SIM_total_p53_conc, 
    _sim_emp_time, _sim_emp_concs,
    PAPER_time, PAPER_relative_total_p53_conc,
    color_data
):
    p = bokeh.plotting.figure(
        height=350, width=650, title="empirical validation of total p53", 
        x_axis_label="time (hrs)", y_axis_label="[ total p53 ] (mM)"
    )
    p.line(SIM_time, SIM_total_p53_conc, line_width=3, line_color="#bebebe")
    p.line(_sim_emp_time, _sim_emp_concs, line_width=3, line_color="black", line_dash="dotdash", )
    p.circle(_sim_emp_time, _sim_emp_concs, size=8, color="black")

    p.line(PAPER_time, PAPER_relative_total_p53_conc, line_width=3, line_color=color_data)
    p.circle(PAPER_time, PAPER_relative_total_p53_conc, size=8, color=color_data)

    legend = bokeh.models.Legend(items=[
        ('numerical',[ p.line(line_width=3, line_color="grey") ]),
        ('simulated', [ p.line(line_width=3, line_color="black", line_dash="dotdash"),
                        p.circle(size=8, color="black")]),
        ('empirical', [ p.line(line_width=3, line_color=color_data), 
                        p.circle(size=8, color=color_data) ])
    ], location="center")
    p.add_layout(legend, 'right')
    p = style(p, autohide=True, toolbar_location='left')
    
    return p
        
        
def simulate_data(SIM_time, SIM_total_p53_conc, PAPER_time):
    _sim_emp_time, _sim_emp_concs = [], []
    for time in PAPER_time: 
        _index = find_index_nearest(SIM_time, time)
        _sim_emp_time.append(SIM_time[_index])
        _sim_emp_concs.append(SIM_total_p53_conc[_index])
    return _sim_emp_time, _sim_emp_concs


# ************************************* NO DECAY CODE *************************************
def B_sustain_single_addition_explorer(
    Xo, τ_n, τ_m, τ_w, args, 
    PAPER_time, PAPER_relative_total_p53_conc, color_data
):
    T_MAX_slider = pn.widgets.IntSlider(name='maximum time', 
        start=12, end=52, step=5, value=47, width=150)
    T_N_STEP_slider = pn.widgets.FloatSlider(name='time of N step', 
        start=0.1, end=6, step=0.1, value=2.8, width=150)
    N_STEP_MAG_slider = pn.widgets.FloatSlider(name='magnitude of N step', 
        start=0, end=10, step=0.1, value=5.1, width=150)
    ZOOM_toggle = pn.widgets.Toggle(name="ZOOM", width=150)
    
    @pn.depends(T_MAX_slider, N_STEP_MAG_slider, T_N_STEP_slider, ZOOM_toggle)
    def _sustain_explorer(T_MAX, N_STEP_MAG, T_N_STEP, ZOOM):
        to = 0
        tf = T_MAX
        timestep = 1000
        t_range = np.linspace(to, tf, timestep)
        
        _index = find_index_nearest(t_range, T_N_STEP)
        N = np.array([0]*_index + [N_STEP_MAG]*(timestep-_index))

        output = trajectory(N, *Xo, to, tf, timestep, τ_n, τ_m, τ_w, args)
        SIM_time, Pi_traj, Pa_traj, M_traj, W_traj, S_traj = output
        SIM_total_p53_conc = Pi_traj + Pa_traj
        
        _sim_emp_time, _sim_emp_concs = simulate_data(SIM_time, SIM_total_p53_conc, PAPER_time)
        
        x_range = (-0.8, 13) if ZOOM else None
        p = bokeh.plotting.figure(
            height=350, width=525, title="single addition of Nutlin-3",
            x_axis_label="time (hrs)", y_axis_label="[ total p53 ] (mM)", x_range=x_range)
            
        # numerical simulation    
        _index_zoom = find_index_nearest(t_range, 12.0)
        SIM_time = SIM_time[:_index_zoom] if ZOOM else SIM_time
        SIM_total_p53_conc = SIM_total_p53_conc[:_index_zoom] if ZOOM else SIM_total_p53_conc
        p.line(SIM_time, SIM_total_p53_conc, 
               line_width=2.5, line_color="black", legend_label="numerical")
        
        # empirical data   
        p.line(PAPER_time, PAPER_relative_total_p53_conc, 
               line_width=3, line_color=color_data, legend_label="empirical")
        p.circle(PAPER_time, PAPER_relative_total_p53_conc, 
               size=8, color=color_data, legend_label="empirical")
               
        # simulated data  
        p.line(_sim_emp_time, _sim_emp_concs, line_width=3, line_color="black", line_dash="dotdash", )
        p.circle(_sim_emp_time, _sim_emp_concs, size=8, color="black")

        p.legend.location = "bottom_right"
        return style(p, toolbar_location='left')

    lay_widgets = pn.Column(
        ZOOM_toggle, T_MAX_slider, T_N_STEP_slider, N_STEP_MAG_slider, align='center')
    return pn.Row(_sustain_explorer, lay_widgets)
    
    
def B_sustain_single_addition_static(
    Xo, τ_n, τ_m, τ_w, args, 
    PAPER_time, PAPER_relative_total_p53_conc, color_data,
    T_MAX, T_N_STEP, N_STEP_MAG, ZOOM=False
):
    to = 0
    tf = T_MAX
    timestep = 1000
    t_range = np.linspace(to, tf, timestep)
    
    _index = find_index_nearest(t_range, T_N_STEP)
    N = np.array([0]*_index + [N_STEP_MAG]*(timestep-_index))

    output = trajectory(N, *Xo, to, tf, timestep, τ_n, τ_m, τ_w, args)
    SIM_time, Pi_traj, Pa_traj, M_traj, W_traj, S_traj = output
    SIM_total_p53_conc = Pi_traj + Pa_traj
    
    _sim_emp_time, _sim_emp_concs = simulate_data(SIM_time, SIM_total_p53_conc, PAPER_time)
    
    x_range = (-0.8, 13) if ZOOM else None
    p = bokeh.plotting.figure(
        height=300, width=425, title=f"N: {N_STEP_MAG} mM",
        x_axis_label="time (hrs)", y_axis_label="[ total p53 ] (mM)", x_range=x_range)
        
    # numerical simulation    
    _index_zoom = find_index_nearest(t_range, 12.0)
    SIM_time = SIM_time[:_index_zoom] if ZOOM else SIM_time
    SIM_total_p53_conc = SIM_total_p53_conc[:_index_zoom] if ZOOM else SIM_total_p53_conc
    p.line(SIM_time, SIM_total_p53_conc, 
           line_width=2.5, line_color="black")
    
    # empirical data   
    p.line(PAPER_time, PAPER_relative_total_p53_conc, 
           line_width=3, line_color=color_data)
    p.circle(PAPER_time, PAPER_relative_total_p53_conc, 
           size=8, color=color_data)
           
    # simulated data  
    p.line(_sim_emp_time, _sim_emp_concs, line_width=3, line_color="black", line_dash="dotdash", )
    p.circle(_sim_emp_time, _sim_emp_concs, size=8, color="black")

    # p.legend.location = "bottom_right"
    return style(p, toolbar_location='left')


def magnitude_effect(ps, qs):
    ZOOM_toggle = pn.widgets.Toggle(name="ZOOM", width=400, value=False)
    
    @pn.depends(ZOOM_toggle)
    def _magnitude_effect(ZOOM):
        plot_ps = bokeh.layouts.gridplot(ps, ncols=2)
        plot_qs = bokeh.layouts.gridplot(qs, ncols=2)
        return plot_qs if ZOOM else plot_ps

    return pn.Column(_magnitude_effect, pn.Row(ZOOM_toggle, align="center"))




# ************************************* DECAY CODE *************************************
def derivs_decay(
    X, t, 
    N_delay, Pa_delay_M, Pa_delay_W, 
    β_p, β_sp, β_m, β_mi, β_w, β_s,
    γ_pi, γ_m, γ_w, γ_s, γ_n,
    δ_mpi, δ_mpa, δ_sm, δ_ws,
    k, k_w, k_s, k_on, k_off,
    n_n, n_s, n_w
):
    Pi, Pa, M, W, S, N, NM = X          # unpacking vector
    
    dPi_dt = ( β_p - γ_pi*Pi 
               - β_sp*Pi * (S/k_s)**n_s/(1+(S/k_s)**n_s) 
               - δ_mpi*M*Pi/(1+(N_delay/k)**n_n) )
    dPa_dt = (   β_sp*Pi * (S/k_s)**n_s/(1+(S/k_s)**n_s) 
               - δ_mpa*M*Pa/(1+(N_delay/k)**n_n) )
    dM_dt = β_mi - γ_m*M + β_m*Pa_delay_M - δ_sm*S*M - k_on*N*M + k_off*NM
    dW_dt =      - γ_w*W + β_w*Pa_delay_W
    dS_dt = β_s - γ_s*S - δ_ws*S*(W/k_w)**n_w/(1+(W/k_w)**n_w)
    dN_dt = -γ_n*N - k_on*N*M + k_off*NM
    dNM_dt = k_on*N*M - k_off*NM
    
    return np.array([dPi_dt, dPa_dt, dM_dt, dW_dt, dS_dt, dN_dt, dNM_dt])


def trajectory_decay_single(
    t_N_step, mag_N_step, dur_N_step, 
    Pi_o, Pa_o, M_o, W_o, S_o, N_o, NM_o,
    to, tf, timestep, 
    args
    
):
    t_range = np.linspace(to, tf, timestep)                 # setting up time array

    _index_N_step = find_index_nearest(t_range, t_N_step)
    _index_N_off = find_index_nearest(t_range, t_N_step + dur_N_step)


    Pi, Pa, M, W, S, N_o, NM_o = Pi_o, Pa_o, M_o, W_o, S_o, N_o, NM_o        # initializing.... 
    Pi_traj, Pa_traj, M_traj, W_traj, S_traj = [Pi_o], [Pa_o], [M_o], [W_o], [S_o]
    N_traj, NM_traj = [N_o], [NM_o]


    for i in range(len(t_range)-1):                    # loop integration
        t = [t_range[i], t_range[i+1]]

        if t[0] < t_N_step: 
            N, NM = 0, 0
        elif t[0] <= t_N_step + dur_N_step:
            N = mag_N_step

        N, NM = floor(N), floor(NM)
        Xo = np.array([Pi, Pa, M, W, S, N, NM])

        N_delay = N_o if t[0] < 0.4 else interpolate(t_range[:i], N_traj[:i], t[0]-0.4)
        Pa_delay_M = Pa_o if t[0] < 0.7 else interpolate(t_range[:i], Pa_traj[:i], t[0]-0.7)
        Pa_delay_W = Pa_o if t[0] < 1.25 else interpolate(t_range[:i], Pa_traj[:i], t[0]-1.25)
        N_delay = floor(N_delay)
        args_all = (N_delay, Pa_delay_M, Pa_delay_W, *args)
        
        _Pi, _Pa, _M, _W, _S, _N, _NM = scipy.integrate.odeint(derivs_decay, Xo, t, args=args_all).T
        i, Pa, M, W, S, N, NM = _Pi[-1], _Pa[-1], _M[-1], _W[-1], _S[-1], _N[-1], _NM[-1]

        Pi, Pa, M, W, S = floor(Pi), floor(Pa), floor(M), floor(W), floor(S)

        Pi_traj.append(Pi)
        Pa_traj.append(Pa)
        M_traj.append(M)
        W_traj.append(W)
        S_traj.append(S)
        N_traj.append(N)
        NM_traj.append(NM)       

    Pi_traj, Pa_traj = np.array(Pi_traj), np.array(Pa_traj)
    M_traj, W_traj, S_traj = np.array(M_traj), np.array(W_traj), np.array(S_traj)
    N_traj, NM_traj = np.array(N_traj), np.array(NM_traj)

    return t_range, Pi_traj, Pa_traj, M_traj, W_traj, S_traj, N_traj, NM_traj
    
def trajectory_decay_multiple(
    t_N_step1, dur_N_step1, mag_N_step1, 
    t_N_step2, dur_N_step2, mag_N_step2,
    t_N_step3, dur_N_step3, mag_N_step3,
    Pi_o, Pa_o, M_o, W_o, S_o, N_o, NM_o,
    to, tf, timestep, 
    args
):
    t_range = np.linspace(to, tf, timestep)                 # setting up time array
    
    Pi, Pa, M, W, S, N_o, NM_o = Pi_o, Pa_o, M_o, W_o, S_o, N_o, NM_o        # initializing.... 
    Pi_traj, Pa_traj, M_traj, W_traj, S_traj = [Pi_o], [Pa_o], [M_o], [W_o], [S_o]
    N_traj, NM_traj = [N_o], [NM_o]

    _first_time_2, _first_time_3 = True, True
    for i in range(len(t_range)-1):                    # loop integration
        t = [t_range[i], t_range[i+1]]

        
        if t[0] < t_N_step1: 
            N, NM = 0, 0
            
        if t_N_step1 <= t[0] < t_N_step1 + dur_N_step1:
            N = mag_N_step1
        
        if (t_N_step2 <= t[0] < t_N_step2 + dur_N_step2) and _first_time_2: 
            ceiling = N + mag_N_step2
            _first_time_2 = False
            
        if (t_N_step2 <= t[0] < t_N_step2 + dur_N_step2):
            N = ceiling
        
        if (t_N_step3 <= t[0] < t_N_step3 + dur_N_step3) and _first_time_3: 
            ceiling = N + mag_N_step3
            _first_time_3 = False
            
        if (t_N_step3 <= t[0] < t_N_step3 + dur_N_step3):
            N = ceiling
            
        
        N, NM = floor(N), floor(NM)
        Xo = np.array([Pi, Pa, M, W, S, N, NM])

        N_delay = N_o if t[0] < 0.4 else interpolate(t_range[:i], N_traj[:i], t[0]-0.4)
        Pa_delay_M = Pa_o if t[0] < 0.7 else interpolate(t_range[:i], Pa_traj[:i], t[0]-0.7)
        Pa_delay_W = Pa_o if t[0] < 1.25 else interpolate(t_range[:i], Pa_traj[:i], t[0]-1.25)
        N_delay = floor(N_delay)
        args_all = (N_delay, Pa_delay_M, Pa_delay_W, *args)
        
        _Pi, _Pa, _M, _W, _S, _N, _NM = scipy.integrate.odeint(derivs_decay, Xo, t, args=args_all).T
        i, Pa, M, W, S, N, NM = _Pi[-1], _Pa[-1], _M[-1], _W[-1], _S[-1], _N[-1], _NM[-1]

        Pi, Pa, M, W, S = floor(Pi), floor(Pa), floor(M), floor(W), floor(S)

        Pi_traj.append(Pi)
        Pa_traj.append(Pa)
        M_traj.append(M)
        W_traj.append(W)
        S_traj.append(S)
        N_traj.append(N)
        NM_traj.append(NM)       

    Pi_traj, Pa_traj = np.array(Pi_traj), np.array(Pa_traj)
    M_traj, W_traj, S_traj = np.array(M_traj), np.array(W_traj), np.array(S_traj)
    N_traj, NM_traj = np.array(N_traj), np.array(NM_traj)

    return t_range, Pi_traj, Pa_traj, M_traj, W_traj, S_traj, N_traj, NM_traj
    
def B_sustain_single_addition_decay_explorer(Xo_decay, args_):
    β_p, β_sp, β_m, β_mi, β_w, β_s, γ_pi, γ_m, γ_w, γ_s, δ_mpi, δ_mpa, δ_sm, δ_ws, k, k_w, k_s, n_n, n_s, n_w = args_

    log_k_nm_slider = pn.widgets.FloatSlider(name='log k_nm',start=-1,end=1,value=0,step=0.1,width=110)
    log_γ_n_slider = pn.widgets.FloatSlider(name='log γ_n',start=-4,end=1,value=-1,step=0.1,width=110)
    mag_N_step_slider = pn.widgets.FloatSlider(name='magnitude N_step',start=1,end=9,value=5,step=0.5,width=110)
    t_N_step_slider = pn.widgets.FloatSlider(name='time N_step',start=1,end=6,value=2.8,step=0.1,width=110)
    dur_N_step_slider = pn.widgets.FloatSlider(name='duration N_step',start=0.5,end=3.5,value=1,step=0.1,width=110)

    @pn.depends(log_k_nm_slider.param.value, log_γ_n_slider.param.value, 
                mag_N_step_slider.param.value, t_N_step_slider.param.value, dur_N_step_slider.param.value)
    def _B_single_addition_decay_explorer(log_k_nm, log_γ_n, mag_N_step, t_N_step, dur_N_step):
        γ_n = np.power(10.0, log_γ_n)
        k_off = 1
        k_on = np.power(10.0, log_k_nm)
        args_decay =  (
            β_p, β_sp, β_m, β_mi, β_w, β_s,
            γ_pi, γ_m, γ_w, γ_s, γ_n,
            δ_mpi, δ_mpa, δ_sm, δ_ws,
            k, k_w, k_s, k_on, k_off,
            n_n, n_s, n_w
        )
        args_step = (t_N_step, mag_N_step, dur_N_step)
        to, tf, timestep = 0, 12, 1000
        output_decay = trajectory_decay_single(*args_step, *Xo_decay, to, tf, timestep, args_decay)
        SIM_time_decay, Pi_traj_decay, Pa_traj_decay, M_traj_decay, _, __, N_traj_decay, ___ = output_decay

        SIM_total_p53_conc_decay = Pi_traj_decay + Pa_traj_decay
        # _sim_emp_time_decay, _sim_emp_concs_decay = simulate_data(SIM_time_decay, SIM_total_p53_conc_decay, PAPER_time)

        p = bokeh.plotting.figure(height=250, width=500, y_axis_label="[total p53] (mM)", x_axis_label="time (hrs)")
        q = bokeh.plotting.figure(height=125, width=500, y_axis_label="[nutlin-3] (mM)",  title="single addition nutlin decay")

        q.line(SIM_time_decay, N_traj_decay, line_width=4, color="#D65E4C")
        p.line(SIM_time_decay, SIM_total_p53_conc_decay, line_width=4, color="black")
        p.line((0, 12), (1, 1), line_width=4, color="#666666", line_dash='dotdash')

        q.xaxis.visible=False
        p, q = style(p, autohide=True), style(q, autohide=True)

        return pn.Column(q, p)
    lay_params = pn.Column(
        log_k_nm_slider, 
        log_γ_n_slider, 
        pn.Spacer(height=25),
        t_N_step_slider, 
        dur_N_step_slider,
        mag_N_step_slider,
        align="center"
    )
    return pn.Row(_B_single_addition_decay_explorer, lay_params)
    
    
    
    
def B_sustain_multiple_addition_decay_explorer(Xo_decay, args_):
    β_p, β_sp, β_m, β_mi, β_w, β_s, γ_pi, γ_m, γ_w, γ_s, δ_mpi, δ_mpa, δ_sm, δ_ws, k, k_w, k_s, n_n, n_s, n_w = args_

    log_k_nm_slider = pn.widgets.FloatSlider(name='log k_nm',start=0,end=2,value=1.5,step=0.1,width=110)
    log_γ_n_slider = pn.widgets.FloatSlider(name='log γ_n',start=-2,end=0,value=-0.5,step=0.1,width=110)

    mag_N_step1_slider = pn.widgets.FloatSlider(name='mag 1',start=1,end=6,value=3,step=0.5,width=85)
    t_N_step1_slider = pn.widgets.FloatSlider(name='time 1',start=0.5,end=3,value=2.9,step=0.1,width=85)
    dur_N_step1_slider = pn.widgets.FloatSlider(name='dur 1',start=0.5,end=1,value=0.8,step=0.1,width=85)

    mag_N_step2_slider = pn.widgets.FloatSlider(name='mag 2',start=4,end=9,value=9,step=0.5,width=85)
    t_N_step2_slider = pn.widgets.FloatSlider(name='time 2',start=3,end=6,value=5.2,step=0.1,width=85)
    dur_N_step2_slider = pn.widgets.FloatSlider(name='dur 2',start=0.5,end=1,value=0.6,step=0.1,width=85)

    mag_N_step3_slider = pn.widgets.FloatSlider(name='mag 3',start=1,end=12,value=12,step=0.5,width=85)
    t_N_step3_slider = pn.widgets.FloatSlider(name='time 3',start=6,end=9,value=7.8,step=0.1,width=85)
    dur_N_step3_slider = pn.widgets.FloatSlider(name='dur 3',start=0.5,end=1,value=1,step=0.1,width=85)

    @pn.depends(log_k_nm_slider.param.value, log_γ_n_slider.param.value, 
                mag_N_step1_slider.param.value, t_N_step1_slider.param.value, dur_N_step1_slider.param.value,
                mag_N_step2_slider.param.value, t_N_step2_slider.param.value, dur_N_step2_slider.param.value,
                mag_N_step3_slider.param.value, t_N_step3_slider.param.value, dur_N_step3_slider.param.value,
    )
    def _B_multiple_addition_decay_explorer(
        log_k_nm, log_γ_n, 
        mag_N_step1, t_N_step1, dur_N_step1,
        mag_N_step2, t_N_step2, dur_N_step2,
        mag_N_step3, t_N_step3, dur_N_step3,
    ):
        γ_n = np.power(10.0, log_γ_n)
        k_off = 1
        k_on = np.power(10.0, log_k_nm)
        args_decay =  (
            β_p, β_sp, β_m, β_mi, β_w, β_s,
            γ_pi, γ_m, γ_w, γ_s, γ_n,
            δ_mpi, δ_mpa, δ_sm, δ_ws,
            k, k_w, k_s, k_on, k_off,
            n_n, n_s, n_w
        )

        args_step = (t_N_step1, dur_N_step1, mag_N_step1,
                     t_N_step2, dur_N_step2, mag_N_step2,
                     t_N_step3, dur_N_step3, mag_N_step3)
        to, tf, timestep = 0, 12, 1000
        
        Pi_o, Pa_o, M_o, W_o, S_o, N_o, NM_o = 0.13, 0, 0.2, 0, 0, 0, 0
        # output_decay = trajectory_decay_multiple(*args_step, *Xo_decay, to, tf, timestep, args_decay)
        output_decay = trajectory_decay_multiple(
            t_N_step1, dur_N_step1, mag_N_step1, 
            t_N_step2, dur_N_step2, mag_N_step2,
            t_N_step3, dur_N_step3, mag_N_step3,
            Pi_o, Pa_o, M_o, W_o, S_o, N_o, NM_o,
            to, tf, timestep, 
            args_decay
        )
        SIM_time_decay, Pi_traj_decay, Pa_traj_decay, M_traj_decay, _, __, N_traj_decay, ___ = output_decay
        SIM_total_p53_conc_decay = Pi_traj_decay + Pa_traj_decay

        p = bokeh.plotting.figure(height=250, width=500, y_axis_label="[total p53] (mM)", x_axis_label="time (hrs)")
        q = bokeh.plotting.figure(height=125, width=500, y_axis_label="[nutlin-3] (mM)",  title="multiple addition nutlin decay")

        q.line(SIM_time_decay, N_traj_decay, line_width=4, color="#D65E4C")
        p.line(SIM_time_decay, SIM_total_p53_conc_decay, line_width=4, color="black")
        p.line((0, 12), (1, 1), line_width=4, color="#666666", line_dash='dotdash')

        q.xaxis.visible=False
        p, q = style(p, autohide=True), style(q, autohide=True)
        return pn.Column(q, p)
        
    lay_params = pn.Column(
        log_k_nm_slider, 
        log_γ_n_slider, 
        pn.Spacer(height=25),
        t_N_step1_slider, 
        dur_N_step1_slider,
        mag_N_step1_slider,
        t_N_step2_slider, 
        dur_N_step2_slider,
        mag_N_step2_slider,
        t_N_step3_slider, 
        dur_N_step3_slider,
        mag_N_step3_slider,
        align="center"
    )
    lay_params = pn.Column(pn.Spacer(height=50), log_k_nm_slider, log_γ_n_slider)
    lay_steps = pn.Row(
        pn.Column(t_N_step1_slider, dur_N_step1_slider, mag_N_step1_slider), 
        pn.Column(t_N_step2_slider, dur_N_step2_slider, mag_N_step2_slider), 
        pn.Column(t_N_step3_slider, dur_N_step3_slider, mag_N_step3_slider), 
        align="center"
    )
    return pn.Row(pn.Column(_B_multiple_addition_decay_explorer, lay_steps), lay_params)
        