import numpy as np
import pandas as pd
import scipy.integrate

import bokeh.io
bokeh.io.output_notebook()
import panel as pn
pn.extension()

def style(p, autohide=False):
    p.title.text_font="Helvetica"
    p.title.text_font_size="16px"
    p.title.align="center"
    p.xaxis.axis_label_text_font="Helvetica"
    p.yaxis.axis_label_text_font="Helvetica"

    p.xaxis.axis_label_text_font_size="14px"
    p.yaxis.axis_label_text_font_size="14px"
    p.xaxis.axis_label_text_font_style = "normal"
    p.yaxis.axis_label_text_font_style = "normal"
    p.background_fill_alpha = 0
    if autohide: p.toolbar.autohide=True
    return p


import scipy.interpolate

def response_time_calculator(t, m, p):
    '''
    Calculates full response behavior. Returns all arrays for plotting.
    Note to self: calculations tricky since domain is not uniform computationally,
        must use scipy.interpolate.interp1d along both axes for good results.
    '''
    _x = np.linspace(m.min(), p.max(), 1000)

    f = scipy.interpolate.interp1d(m, t)
    t_m = f(_x)

    f = scipy.interpolate.interp1d(p, t)
    t_p = f(_x)

    t_response = t_p - t_m

    filtered_t_response = []
    for _t in t_response:
        if _t < 0: break
        filtered_t_response.append(_t)
    filtered_t_response = np.array(filtered_t_response)
    return _x, t_m, t_p, t_response, filtered_t_response[:-1]

def half_steady_state(arr, val, times, return_index=False):
    '''
    Return t1/2. This value is nice summary for when x = k.
    Note to self: this is a bit reductive, and we'll soon see intersections in
        the full response plot. This indicates
    '''
    if type(arr)!= np.ndarray:
        arr = np.array(arr)
    _arr = np.abs(arr - val)                           # minimum element closest to val
    index = list(_arr).index(np.min(_arr))             # locate index of minimum element
    t_steady = times[index]
    if return_index: return index
    return t_steady

def deriv_unregulated(mp, t, γ):
    m, p = mp
    m_deriv = 1-m
    p_deriv = γ*(m-p)
    return np.array([m_deriv, p_deriv])

def deriv_regulated(mp, t, γ, κ, n):
    m, p = mp
    m_deriv = 1/((1+p/κ)**n) - m
    p_deriv = γ*(m-p)
    return np.array([m_deriv, p_deriv])

palette = ["#31a354",
           "#74c476", # "#a1d99b"
           "#e6550d",
           "#fd8d3c",]

css = '''
.bk.panel-font {
    font-family: "Open Sans"
}
'''

pn.extension(raw_css=[css])


log_γ_slider = pn.widgets.FloatSlider(start=-1, end=2.5, value=0, name="log γ", width=250,step=0.1)
log_κ_slider = pn.widgets.FloatSlider(start=-5, end=5, value=0, name="log κ", width=250,step=0.5)
n_slider = pn.widgets.FloatSlider(start=0, end=10, value=1, name="n", width=250,)

normalize_checkbox = pn.widgets.Checkbox(name="normalize")
response_half_checkbox = pn.widgets.Checkbox(name="half-steady response time", value=True)
response_full_checkbox = pn.widgets.Checkbox(name="full response analysis", value=True)
shade_checkbox = pn.widgets.Checkbox(name="shade trace (if full response on)")
shade_response_checkbox = pn.widgets.Checkbox(name="shade response (if full response on)")

t_slider = pn.widgets.RangeSlider(name="time", start=0.0, end=50, value=(0, 10,), step=5, width=500)

@pn.depends(log_γ_slider.param.value, log_κ_slider.param.value, n_slider.param.value,
            normalize_checkbox.param.value,
            response_full_checkbox.param.value, response_half_checkbox.param.value,
            shade_checkbox.param.value, shade_response_checkbox.param.value,
            t_slider.param.value
           )
def plotter(log_γ=0, log_κ=0, n=1, normalize=False,
            response_full=True, response_half=True,
            shade=False, shade_response=False,
            t_range=(0, 10)
           ):

    # .... INITIALIZING ....
    γ, κ = 10**log_γ, 10**log_κ
    mp0 = np.array([0.0, 0.0])
    t = np.linspace(t_range[0], t_range[1], 750)

    # .... INTEGRATING ....
    mp_unregulated = scipy.integrate.odeint(deriv_unregulated, mp0, t, args=(γ,))   # NEEDS A COMMA!!!
    m_unregulated, p_unregulated = mp_unregulated.T

    mp_regulated = scipy.integrate.odeint(deriv_regulated, mp0, t, args=(γ, κ, n))
    m_regulated, p_regulated = mp_regulated.T

    # .... NORMALIZING ....
    if normalize:
        m_unregulated /= m_unregulated.max()
        p_unregulated /= p_unregulated.max()
        m_regulated /= m_regulated.max()
        p_regulated /= p_regulated.max()

    # .... RESPONSE TIMES ....
    t_st_m_unregulated = half_steady_state(m_unregulated, 0.5, t)
    t_st_p_unregulated = half_steady_state(p_unregulated, 0.5, t)
    t_st_diff_unregulated = t_st_p_unregulated - t_st_m_unregulated

    _reg_val = (m_regulated[-1])/2.0
    t_st_m_regulated = half_steady_state(m_regulated, _reg_val, t)
    t_st_p_regulated = half_steady_state(p_regulated, _reg_val, t)
    t_st_diff_regulated = t_st_p_regulated - t_st_m_regulated

    # .... PLOTTING ....
    p = bokeh.plotting.figure(
        title="negative autoregulation",
        x_axis_label="nondimensionalized time",
        y_axis_label=f"{'normalized ' if normalize else ''}nondimensionalized [m], [p]"
    )
    for arr, label, color in zip([m_unregulated, p_unregulated, m_regulated, p_regulated],
                                 ["m unregulated", "p unregulated", "m regulated", "p regulated"],
                                 palette):
            p.line(t, arr, color=color, legend_label=label, line_width=2.4)

    if response_half:
        p.line((t_st_m_unregulated, t_st_p_unregulated),(0.5, 0.5),
               color=palette[1], line_width=2)
        p.line((t_st_m_regulated, t_st_p_regulated),(_reg_val, _reg_val),
               color=palette[3], line_width=2)

        p.circle(t_st_m_unregulated, 0.5, color=palette[0], size=8)
        p.circle(t_st_p_unregulated, 0.5, color=palette[1], size=8)
        p.circle(t_st_m_regulated, _reg_val, color=palette[2], size=8)
        p.circle(t_st_p_regulated, _reg_val, color=palette[3], size=8)

    p.legend.location = "bottom_right"
    p.legend.click_policy = 'hide'

    # *********************************** FULL RESPONSE ANALYSIS ***********************************
    if response_full:

        # .... RETRIEVING ....
        response_reg = response_time_calculator(t, m_regulated, p_regulated)
        _x_reg, t_m_reg, t_p_reg , t_response_reg, filtered_t_response_reg = response_reg

        response_unreg = response_time_calculator(t, m_unregulated, p_unregulated)
        _x_unreg, t_m_unreg, t_p_unreg , t_response_unreg, filtered_t_response_unreg = response_unreg

        _pt_unreg = half_steady_state(filtered_t_response_unreg, t_st_diff_unregulated, _x_unreg, return_index=True)
        _pt_reg = half_steady_state(filtered_t_response_reg, t_st_diff_regulated, _x_reg, return_index=True)

        # .... PLOTTING ....
        q = bokeh.plotting.figure(width=450, height=490,
            title="response time: p - m",
            y_axis_label=f"{'normalized 'if normalize else ''}nondimensionalized time difference"
        )
        q.line(np.arange(0, len(filtered_t_response_unreg)), filtered_t_response_unreg,
               color="darkgreen", line_width=3, line_dash="dotdash", legend_label="unregulated")
        q.line(np.arange(0, len(filtered_t_response_reg)), filtered_t_response_reg,
               color="orangered", line_width=3, line_dash="dashdot", legend_label="regulated")

        if response_half:
            q.circle(_pt_unreg, t_st_diff_unregulated, color="darkgreen", size=8)
            q.circle(_pt_reg, t_st_diff_regulated, color="orangered", size=8)

        q.xaxis.visible, q.xgrid.visible = False, False
        q.legend.location = "bottom_right"
        q.outline_line_color = None

        # .... MARKDOWN TABLE ....
        # only integrate up until regulated for comparitive purposes
        integral_unreg = np.trapz(filtered_t_response_unreg[:len(filtered_t_response_reg)],
                                  dx=_x_unreg[1]-_x_unreg[0])
        integral_reg = np.trapz(filtered_t_response_reg, dx=_x_unreg[1]-_x_unreg[0])
        _header2 = pn.panel("All fractional times", align="center",
                            style={"font-size": '15px',"font-family":"Open Sans","font-style":"italic"})
        md_table = pn.panel(f"""
                            | system | integrated ΔT |
                            |:-----------:|-----------:|
                            | unreg | {np.round(integral_unreg, 4)} |
                            | reg | {np.round(integral_reg, 4)} |
                            | {'DIFF' if shade_response else 'diff'} | {np.round(integral_unreg-integral_reg, 4)} |
                            """,
                            align="center",
                            style={"font-family":"Open Sans"}
                           )
        _header1 = pn.panel("Half steady-state time", align="center",
                            style={"font-size": '15px',"font-family":"Open Sans", "font-style":"italic"})
        table_response = pn.pane.Markdown(f"""
                        | system      |      m     |      p     |    p-m    |
                        |:-----------:|-----------:|-----------:|-----------:|
                        | unreg | {np.round(t_st_m_unregulated,3)} | \
                            {np.round(t_st_p_unregulated, 3)} | \
                            {np.round(t_st_diff_unregulated, 3)}
                        | reg | {np.round(t_st_m_regulated, 3)} | \
                            {np.round(t_st_p_regulated, 3)} | \
                            {np.round(t_st_diff_regulated, 3)}
                        | diff | {np.round(t_st_m_unregulated - t_st_m_regulated, 3)} | \
                                {np.round(t_st_p_unregulated - t_st_p_regulated, 3)} | \
                                {np.round(t_st_diff_unregulated - t_st_diff_regulated, 3)}
                        """, align="center",
                    style={'font-family': "Open Sans"})

        # .... VISUALIZE SHADING ....
        if shade:
            for _x, t_tuple, color in zip([_x_reg[:len(filtered_t_response_reg)],
                                           _x_unreg[:len(filtered_t_response_unreg)]],
                                          [(t_p_reg[:len(filtered_t_response_reg)], t_m_reg[:len(filtered_t_response_reg)]),
                                           (t_p_unreg[:len(filtered_t_response_unreg)], t_m_unreg[:len(filtered_t_response_unreg)])],
                                          ["sandybrown", "darkseagreen"] ):
                p.multi_line(xs=[(tp, tm) for tp, tm in zip(*t_tuple)],
                             ys=[(_, _) for _ in _x], line_width=0.2, color=color, line_alpha=0.5)

                # bringing attention to curves
                q.line(np.arange(0, len(filtered_t_response_unreg)), filtered_t_response_unreg,
                       color="darkgreen", line_width=4, line_dash="dotdash", legend_label="unregulated")
                q.line(np.arange(0, len(filtered_t_response_reg)), filtered_t_response_reg,
                       color="orangered", line_width=4, line_dash="dashdot", legend_label="regulated")

        if shade_response:
            # highlight integrated difference
            q.multi_line(xs=[(_, _) for _ in np.arange(0,
                                min(len(filtered_t_response_unreg), len(filtered_t_response_reg))
                            )],
                         ys = [(_t_reg, _t_unreg) for _t_reg, _t_unreg in zip(filtered_t_response_reg,
                                                                              filtered_t_response_unreg)],
                         color="grey", line_width=0.05)

        lay_tables = pn.Row(pn.Spacer(width=45),
                            pn.Column(_header1, table_response),
                            pn.Column(_header2, md_table))
        return pn.Row(style(p), pn.Column(pn.Spacer(height=5), style(q), pn.Spacer(height=10), lay_tables))

    else:
        return style(p)

lay_widgets = pn.Column(pn.Spacer(height=100), log_γ_slider, log_κ_slider, n_slider,
                    pn.Spacer(height=10),
                    pn.Row(pn.Spacer(width=25), normalize_checkbox),
                    pn.Row(pn.Spacer(width=25), response_half_checkbox),
                    pn.Row(pn.Spacer(width=25), response_full_checkbox),
                    pn.Row(pn.Spacer(width=25), shade_checkbox),
                    pn.Row(pn.Spacer(width=25), shade_response_checkbox),
                   css_classes=["panel-font"]
                       )
lay_time = pn.Row(pn.Spacer(width=50), t_slider)
lay = pn.Row(plotter, lay_widgets)
dashboard = pn.Column(lay, lay_time)
dashboard.servable()
