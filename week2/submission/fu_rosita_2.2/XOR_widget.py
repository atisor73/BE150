import numpy as np
import pandas as pd
import biocircuits
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

    p.xaxis.axis_label_text_font_size="13px"
    p.yaxis.axis_label_text_font_size="13px"
    p.background_fill_alpha = 0
    if autohide: p.toolbar.autohide=True
    return p


def derivs(ABZ, t, t_step, gamma_array, beta_array, n_array, step):
    A, B, Z = ABZ
    if t < t_step:
        X = step[0]
        Y = step[1]
    if t >= t_step:
        X = step[2]
        Y = step[3]

    gammaA, gammaB, gammaZ = gamma_array
    betaA, betaB, betaZ = beta_array
    nXA, nXB, nYA, nYB, nAZ, nBZ = n_array

    deriv_A = betaA*(X**nXA*(1-Y**nYA))/((1+X**nXA)*(1+Y**nYA)) - gammaA*A
    deriv_B = betaB*(Y**nYB*(1-X**nXB))/((1+X**nXB)*(1+Y**nYB)) - gammaB*B
    deriv_Z = betaZ*(A**nAZ + B**nBZ)/((1+A**nAZ)*(1+B**nBZ)) - gammaZ*Z

    return np.array([deriv_A, deriv_B, deriv_Z])

gammaA_slider = pn.widgets.FloatSlider(name="γ_A", start=0.1, end=5.0, value=1.0, width=100)
gammaB_slider = pn.widgets.FloatSlider(name="γ_B", start=0.1, end=5.0, value=1.0, width=100)
gammaZ_slider = pn.widgets.FloatSlider(name="γ_Z", start=0.1, end=5.0, value=1.0, width=100)

betaA_slider = pn.widgets.FloatSlider(name="β_A", start=0.1, end=10.0, value=1.0, width=100)
betaB_slider = pn.widgets.FloatSlider(name="β_B", start=0.1, end=10.0, value=1.0, width=100)
betaZ_slider = pn.widgets.FloatSlider(name="β_Z", start=0.1, end=10.0, value=1.0, width=100)

nXA_slider = pn.widgets.FloatSlider(name="n_XA", start=0.1, end=15.0, value=5.0, width=100)
nXB_slider = pn.widgets.FloatSlider(name="n_XB", start=0.1, end=15.0, value=5.0, width=100)
nYA_slider = pn.widgets.FloatSlider(name="n_YA", start=0.1, end=15.0, value=5.0, width=100)
nYB_slider = pn.widgets.FloatSlider(name="n_YB", start=0.1, end=15.0, value=5.0, width=100)
nAZ_slider = pn.widgets.FloatSlider(name="n_AZ", start=0.1, end=15.0, value=5.0, width=100)
nBZ_slider = pn.widgets.FloatSlider(name="n_BZ", start=0.1, end=15.0, value=5.0, width=100)

t_step_slider = pn.widgets.FloatSlider(name="step time", start=0.0, end=5.0, value=3.6, width=300)
Xo_button = pn.widgets.RadioButtonGroup(name="Xo", options=["Xo LOW","Xo HIGH"], value="Xo LOW")
Yo_button = pn.widgets.RadioButtonGroup(name="Yo", options=["Yo LOW","Yo HIGH"], value="Yo LOW")

Xf_button = pn.widgets.RadioButtonGroup(name="Xf", options=["Xf LOW","Xf HIGH"], value="Xf HIGH")
Yf_button = pn.widgets.RadioButtonGroup(name="Yf", options=["Yf LOW","Yf HIGH"], value="Yf HIGH")

normalize_button = pn.widgets.RadioButtonGroup(
    name="normalize", options=["NORMALIZE HALF", "NORMALIZE OFF"], value="NORMALIZE HALF")



@pn.depends(gammaA_slider.param.value, gammaB_slider.param.value, gammaZ_slider.param.value,
            betaA_slider.param.value, betaB_slider.param.value, betaZ_slider.param.value,
            nXA_slider.param.value, nXB_slider.param.value, nYA_slider.param.value,
            nYB_slider.param.value, nAZ_slider.param.value, nBZ_slider.param.value,
            Xo_button.param.value, Yo_button.param.value, Xf_button.param.value, Yf_button.param.value,
            t_step_slider.param.value, normalize_button.param.value
           )
def plotter(gammaA, gammaB, gammaZ,
            betaA, betaB, betaZ,
            nXA, nXB, nYA, nYB, nAZ, nBZ,
            Xo_type, Yo_type, Xf_type, Yf_type,
            t_step, normalize
           ):
    Xo, Yo, Xf, Yf = 0, 0, 0, 0
    if Xo_type == "Xo HIGH": Xo = 1
    if Yo_type == "Yo HIGH": Yo = 1
    if Xf_type == "Xf HIGH": Xf = 1
    if Yf_type == "Yf HIGH": Yf = 1

    ABZo = np.array([0.0, 0.0, 0.0])
    t = np.linspace(0, 10, 500)

    gamma_array = np.array([gammaA, gammaB, gammaZ])
    beta_array = np.array([betaA, betaB, betaZ])
    n_array = np.array([nXA, nXB, nYA, nYB, nAZ, nBZ])
    step = [Xo, Yo, Xf, Yf]
    args = (t_step, gamma_array, beta_array, n_array, step)

    # .... integrating ....
    ABZ = scipy.integrate.odeint(derivs, ABZo, t, args=args)
    A, B, Z = ABZ.T
    if normalize == "NORMALIZE HALF":
        if len(A[A==0]) != len(A): A /= (A.max()*2)
        if len(B[B==0]) != len(B): B /= (B.max()*2)
        if len(Z[Z==0]) != len(Z): Z /= (Z.max()*2)
    X, Y = np.empty(len(t)), np.empty(len(t))
    X[t < t_step] = Xo
    X[t >= t_step] = Xf
    Y[t < t_step] = Yo
    Y[t >= t_step] = Yf

    p = bokeh.plotting.figure(height=400, width=610, title="X[PL]OR",
                              x_axis_label="time", y_axis_label="[ ]")
    p.line(t, X, line_width=3, color="red", legend_label="X",line_dash="dashdot")
    p.line(t, Y, line_width=3, color="orangered", legend_label="Y", line_dash="dotdash")

    p.line(t, A, line_width=3, color="#eba8b5", legend_label="A")
    p.line(t, B, line_width=3, color="#9fc0c1", legend_label="B")
    p.line(t, Z, line_width=3, color="#65042d", legend_label="Z")

    p.legend.click_policy="hide"
    p.legend.location="top_left"


    return style(p)



lay_BG = pn.Column(
            pn.Spacer(height=11),
            pn.Row(gammaA_slider, gammaB_slider, gammaZ_slider, align="center"),
            pn.Row(betaA_slider, betaB_slider, betaZ_slider, align="center")
        )
lay_N = pn.Column(
            pn.Row(nXA_slider, nXB_slider),
            pn.Row(nYA_slider, nYB_slider),
            pn.Row(nAZ_slider, nBZ_slider)
        )
lay_params = pn.Row(lay_BG, lay_N)

lay_time = pn.Row(t_step_slider, align="center")
lay_init = pn.Column(pn.Row(Xo_button, Xf_button),
                     pn.Row(Yo_button, Yf_button))
lay_norm = pn.Column(normalize_button, align="center")
dashboard = pn.Column(lay_params, plotter, lay_time, lay_init, lay_norm)

dashboard.servable()
