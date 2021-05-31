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


def derivs(XYZ, t, t_step, gamma_array, beta_array):
    X, Y, Z = XYZ
    gammaX, gammaY, gammaZ = gamma_array
    betaX, betaY, betaZ = beta_array
    nXZ, nZX, nYZ, nZY, nΧΧ, nΥΥ = n_array

    deriv_X = betaX*(Z**nZX*(1-X**nZX))/((1+X**nXX)*(1+Z**nZX)) - gammaX*X
    deriv_Y = betaY*(Z**nZY*(1-Y**nYY))/((1+Y**nYY)*(1+Z**nZY)) - gammaY*Y
    deriv_Z = betaZ*(X**nXZ + Y**nYZ)/((1+X**nXZ)*(1+Y**nYZ)) - gammaZ*Z

    return np.array([deriv_X, deriv_Y, deriv_Z])

gammaX_slider = pn.widgets.FloatSlider(name="γ_X", start=0.1, end=5.0, value=1.0, width=100)
gammaY_slider = pn.widgets.FloatSlider(name="γ_Y", start=0.1, end=5.0, value=1.0, width=100)
gammaZ_slider = pn.widgets.FloatSlider(name="γ_Z", start=0.1, end=5.0, value=1.0, width=100)

betaX_slider = pn.widgets.FloatSlider(name="β_X", start=0.1, end=10.0, value=1.0, width=100)
betaY_slider = pn.widgets.FloatSlider(name="β_Y", start=0.1, end=10.0, value=1.0, width=100)
betaZ_slider = pn.widgets.FloatSlider(name="β_Z", start=0.1, end=10.0, value=1.0, width=100)

nXZ_slider = pn.widgets.FloatSlider(name="n_XZ", start=0.1, end=15.0, value=5.0, width=100)
nZX_slider = pn.widgets.FloatSlider(name="n_ZX", start=0.1, end=15.0, value=5.0, width=100)
nYZ_slider = pn.widgets.FloatSlider(name="n_YZ", start=0.1, end=15.0, value=5.0, width=100)
nZY_slider = pn.widgets.FloatSlider(name="n_ZY", start=0.1, end=15.0, value=5.0, width=100)
nXX_slider = pn.widgets.FloatSlider(name="n_XX", start=0.1, end=15.0, value=5.0, width=100)
nYY_slider = pn.widgets.FloatSlider(name="n_YY", start=0.1, end=15.0, value=5.0, width=100)

normalize_button = pn.widgets.RadioButtonGroup(
    name="normalize", options=["NORMALIZE HALF", "NORMALIZE OFF"], value="NORMALIZE HALF")

gammaX = 1
gammaY = 1
gammaZ = 1
beta_X = 1
beta_Y = 1
beta_Z = 1
nXZ = 5
nZX = 5
nYZ = 5
nZY = 5
nΧΧ = 5
nΥΥ = 5

xyzo = np.array([1.0, 0.0, 0.0])
t = np.linspace(0, 10, 500)

# gamma_array = np.array([gammaΧ, gammaY, gammaZ])
# beta_array = np.array([betaΧ, betaΥ, betaZ])
gamma_array = np.array([1, 1, 1])
beta_array = np.array([1, 1, 1])
n_array = np.array([5, 5, 5, 5, 5, 5])

# .... integrating ....
args = (gamma_array, beta_array, n_array)

ΧΥΖ = scipy.integrate.odeint(derivs, xyzo, t, args=args)
Χ, Υ, Ζ = XYZ.T

p = bokeh.plotting.figure(height=400, width=610, title="X[PL]OR",
                          x_axis_label="time", y_axis_label="[ ]")
p.line(t, X, line_width=3, color="#eba8b5", legend_label="X")
p.line(t, Y, line_width=3, color="#9fc0c1", legend_label="Y")
p.line(t, Z, line_width=3, color="#65042d", legend_label="Z")

p.legend.click_policy="hide"
p.legend.location="top_left"
layout = style(p)

layout.servable()
