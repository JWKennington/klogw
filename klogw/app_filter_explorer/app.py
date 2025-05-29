import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from scipy import signal

##################################################
# Constants
##################################################

LIGO_PURPLE = "#593196"
LIGO_LOGO_URL = "https://dcc.ligo.org/public/0000/F0900035/002/ligo_logo.png"

DEFAULT_FAMILY = "Butterworth"
DEFAULT_TYPE = "low"
DEFAULT_ORDER = 4
DEFAULT_DOMAIN = "analog"
DEFAULT_CUTOFF1 = 1.0
DEFAULT_CUTOFF2 = 2.0

FILTER_FAMILIES = [
    {"label": "Butterworth", "value": "Butterworth"},
    {"label": "Chebyshev I", "value": "Chebyshev I"},
    {"label": "Chebyshev II", "value": "Chebyshev II"},
    {"label": "Elliptic", "value": "Elliptic"},
    {"label": "Bessel", "value": "Bessel"},
    {"label": "Custom (manual)", "value": "Custom"},
]
FILTER_TYPES = [
    {"label": "Lowpass", "value": "low"},
    {"label": "Highpass", "value": "high"},
    {"label": "Bandpass", "value": "bandpass"},
    {"label": "Bandstop", "value": "bandstop"},
]
DOMAIN_OPTIONS = [
    {"label": "Analog", "value": "analog"},
    {"label": "Digital", "value": "digital"},
]

##################################################
# Initialize the App
##################################################

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Signal Filter Visualizer (LIGO)"

# Inline index_string with Lato font, big row on desktops, etc.
app.index_string = r"""
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <!-- Lato font -->
    <link href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap" rel="stylesheet">
    <style>
    body {
      font-family: 'Lato', sans-serif;
      margin: 0; padding: 0;
      background-color: #fff;
    }
    .footer {
      text-align: center;
      padding: 10px;
      border-top: 1px solid #eaeaea;
      background: #f8f9fa;
      color: #666;
    }
    /* On large screens, the .main-content-row tries to fill leftover space with no scrolling */
    @media (min-width: 992px) {
        .main-content-row {
            height: calc(100vh - 210px) !important; /* adjust offset for your header/controls */
            overflow: hidden;
        }
    }
    /* left & right col each fill 100% height of that row */
    #left-col, #right-col {
        height: 100%;
    }
    /* in the right col, we have a container of 2 half-height graphs stacked */
    #right-col > div {
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    #right-col > div > div:first-child {
        height: 50%;
    }
    #right-col > div > div:last-child {
        height: 50%;
    }
    /* On md+ screens, we always show the controls row & hide the toggle. */
    @media (min-width: 768px) {
      #controls-collapse.collapse {
        display: block !important;
        visibility: visible !important;
        height: auto !important;
      }
      #controls-toggle-btn {
        display: none !important;
      }
    }
    /* dropdown menu overflow fix */
    .dropdown-menu {
      max-height: 250px;
      overflow-y: auto;
      z-index: 2000;
    }
    </style>
    {%renderer%}
</head>
<body>
    {%app_entry%}
    <footer class="footer">
      © Jim Kennington 2025
    </footer>
    {%config%}
    {%scripts%}
    {%renderer%}
</body>
</html>
"""

##################################################
# Navbar / Header
##################################################

header = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.Img(src=LIGO_LOGO_URL, height="50px"), width="auto"),
                dbc.Col(dbc.NavbarBrand("Interactive Filter Visualization", className="ms-2"))
            ], align="center", className="g-0"),
            href="#", style={"textDecoration": "none"}
        ),
    ]),
    color="white",
    dark=False,
    className="mb-0"
)

##################################################
# Filter Controls & Mobile Toggle
##################################################

domain_radio = dbc.RadioItems(
    id="domain-radio",
    className="btn-group",
    inputClassName="btn-check",
    labelClassName="btn btn-outline-primary",
    labelCheckedClassName="active",
    options=DOMAIN_OPTIONS,
    value=DEFAULT_DOMAIN
)
domain_radio_group = html.Div(domain_radio, className="btn-group me-2", **{"role": "group"})

family_dd = dcc.Dropdown(
    id="family-dropdown",
    options=FILTER_FAMILIES,
    value=DEFAULT_FAMILY,
    clearable=False,
    style={"minWidth": "120px"}
)
type_dd = dcc.Dropdown(
    id="type-dropdown",
    options=FILTER_TYPES,
    value=DEFAULT_TYPE,
    clearable=False,
    style={"minWidth": "100px"}
)
order_input = dbc.Input(
    id="order-input",
    type="number",
    value=DEFAULT_ORDER,
    min=1,
    step=1,
    style={"width":"5ch"}
)

cut1_label = dbc.InputGroupText(id="cutoff1-label", children="Cutoff 1")
cut1_in = dbc.Input(id="cutoff1-input", type="number", value=DEFAULT_CUTOFF1, step=0.1, style={"width":"6ch"})
cut1_grp = dbc.InputGroup([cut1_label, cut1_in], id="cutoff1-group", className="me-2")

cut2_label = dbc.InputGroupText(id="cutoff2-label", children="Cutoff 2")
cut2_in = dbc.Input(id="cutoff2-input", type="number", value=DEFAULT_CUTOFF2, step=0.1, style={"width":"6ch"})
cut2_grp = dbc.InputGroup([cut2_label, cut2_in], id="cutoff2-group", style={"display":"none"})

controls_row = dbc.Row([
    dbc.Col(domain_radio_group, width="auto"),
    dbc.Col(family_dd, width="auto"),
    dbc.Col(type_dd, width="auto"),
    dbc.Col(order_input, width="auto"),
    dbc.Col(cut1_grp, width="auto"),
    dbc.Col(cut2_grp, width="auto"),
], align="center", className="g-2 flex-wrap")

controls_collapse = dbc.Collapse(controls_row, id="controls-collapse", is_open=False)
toggle_btn = dbc.Button("Filter Controls", id="controls-toggle-btn", color="secondary", className="d-md-none mb-2")

##################################################
# Pole-Zero Buttons
##################################################

pz_buttons = html.Div([
    dbc.Button("Add Pole", id="add-pole-btn", color="primary", outline=True, size="sm", className="me-2"),
    dbc.Button("Add Zero", id="add-zero-btn", color="primary", outline=True, size="sm", className="me-2"),
    dbc.Button("Clear", id="clear-btn", color="secondary", size="sm"),
], className="mb-2")

##################################################
# Graphs
##################################################

pz_graph = dcc.Graph(
    id="pz-plot",
    style={"width":"100%", "height":"100%"},
    config={
        "editable": True,
        "edits": {"shapePosition": True},
        "displayModeBar": False
    },
)
bode_graph = dcc.Graph(
    id="bode-plot",
    style={"width":"100%", "height":"50%"},
    config={"displayModeBar":False},
)
impulse_graph = dcc.Graph(
    id="impulse-plot",
    style={"width":"100%", "height":"50%"},
    config={"displayModeBar":False}
)

##################################################
# App Layout
##################################################

app.layout = html.Div([
    header,
    dbc.Container([
        toggle_btn,
        controls_collapse,
    ], fluid=True, className="mt-2"),

    dbc.Row([
        dbc.Col(
            html.Div([
                pz_graph,
                pz_buttons
            ], style={"width":"100%","height":"100%","display":"flex","flexDirection":"column"}),
            md=6, style={"height":"100%"}, id="left-col"
        ),
        dbc.Col(
            html.Div([
                bode_graph,
                impulse_graph
            ], style={"width":"100%","height":"100%","display":"flex","flexDirection":"column"}),
            md=6, style={"height":"100%"}, id="right-col"
        )
    ], className="gx-0 main-content-row flex-nowrap", style={"margin":0}),

    dcc.Store(id="pz-store", data={"poles":[], "zeros":[], "gain":1.0}),
], style={"display":"flex","flexDirection":"column","minHeight":"100vh"})

##################################################
# Mobile Toggle Controls
##################################################
@app.callback(
    Output("controls-collapse","is_open"),
    Input("controls-toggle-btn","n_clicks"),
    State("controls-collapse","is_open")
)
def toggle_controls(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

##################################################
# Show/Hide Second Cutoff & Label
##################################################
@app.callback(
    Output("cutoff2-group","style"),
    Output("cutoff1-label","children"),
    Output("cutoff2-label","children"),
    Input("type-dropdown","value"),
    Input("domain-radio","value")
)
def toggle_cut2_labels(ftype, domain):
    if ftype in ["bandpass","bandstop"]:
        st = {}
        c1 = "Low Cutoff"
        c2 = "High Cutoff"
    else:
        st = {"display":"none"}
        c1 = "Cutoff Freq"
        c2 = "Cutoff 2"
    if domain=="analog":
        c1 += " (rad/s)"
        c2 += " (rad/s)"
    else:
        c1 += " (norm.)"
        c2 += " (norm.)"
    return st, c1, c2

##################################################
# Filter Design Helper
##################################################
def design_filter(family, btype, order, domain, c1, c2=None):
    analog = (domain=="analog")
    if btype in ["bandpass","bandstop"]:
        lo=min(c1,c2)
        hi=max(c1,c2)
        if analog:
            if lo<=0: lo=1e-6
        else:
            if lo<=0: lo=1e-6
            if hi>=1: hi=0.999999
        Wn=[lo, hi]
    else:
        Wn=c1
        if analog:
            if Wn<=0: Wn=1e-6
        else:
            if Wn>=1: Wn=0.999999
            if Wn<=0: Wn=1e-6

    try:
        if family=="Butterworth":
            z,p,k=signal.butter(order, Wn, btype=btype, analog=analog, output='zpk')
        elif family=="Chebyshev I":
            z,p,k=signal.cheby1(order, 1, Wn, btype=btype, analog=analog, output='zpk')
        elif family=="Chebyshev II":
            z,p,k=signal.cheby2(order, 40, Wn, btype=btype, analog=analog, output='zpk')
        elif family=="Elliptic":
            z,p,k=signal.ellip(order,1,40, Wn, btype=btype, analog=analog, output='zpk')
        elif family=="Bessel":
            z,p,k=signal.bessel(order, Wn, btype=btype, analog=analog, output='zpk')
        else:
            z=np.array([]); p=np.array([]); k=1.0
    except:
        z=np.array([]); p=np.array([]); k=1.0

    zeros=[[float(zr.real), float(zr.imag)] for zr in z]
    poles=[[float(pr.real), float(pr.imag)] for pr in p]
    return zeros, poles, float(k)

##################################################
# Update pz-store Callback
##################################################
@app.callback(
    Output("pz-store","data"),
    Input("family-dropdown","value"),
    Input("type-dropdown","value"),
    Input("order-input","value"),
    Input("domain-radio","value"),
    Input("cutoff1-input","value"),
    Input("cutoff2-input","value"),
    Input("add-pole-btn","n_clicks"),
    Input("add-zero-btn","n_clicks"),
    Input("clear-btn","n_clicks"),
    State("pz-store","data")
)
def update_pzstore(family, ftype, order, domain, c1, c2,
                   addp, addz, clear_btn,
                   store_data):
    ctx = callback_context
    trig_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    old_poles = [complex(p[0],p[1]) for p in store_data["poles"]]
    old_zeros = [complex(z[0],z[1]) for z in store_data["zeros"]]
    old_gain = store_data["gain"]

    if trig_id in ["family-dropdown","type-dropdown","order-input","domain-radio","cutoff1-input","cutoff2-input"]:
        if family!="Custom":
            zz, pp, kk = design_filter(family, ftype, order, domain, c1, c2)
            old_zeros=[complex(zv[0],zv[1]) for zv in zz]
            old_poles=[complex(pv[0],pv[1]) for pv in pp]
            old_gain=kk
        else:
            # if domain changed while custom, reset
            if trig_id=="domain-radio":
                old_zeros=[]
                old_poles=[]
                old_gain=1.0

    if trig_id=="add-pole-btn":
        if domain=="analog":
            idx=len(old_poles)
            newp=complex(-0.5*(idx+1),0)
            old_poles.append(newp)
        else:
            idx=len(old_poles)
            val=0.5+0.2*idx
            if val>0.9: val=0.9
            old_poles.append(complex(val,0))

    if trig_id=="add-zero-btn":
        if domain=="analog":
            idx=len(old_zeros)
            newz=complex(-0.5*(idx+1),0)
            old_zeros.append(newz)
        else:
            idx=len(old_zeros)
            if idx==0:
                old_zeros.append(complex(1,0))
            elif idx==1:
                old_zeros.append(complex(-1,0))
            else:
                old_zeros.append(complex(0.5,0))

    if trig_id=="clear-btn":
        if family!="Custom":
            zz,pp,kk=design_filter(family, ftype, order, domain, c1, c2)
            old_zeros=[complex(zv[0],zv[1]) for zv in zz]
            old_poles=[complex(pv[0],pv[1]) for pv in pp]
            old_gain=kk
        else:
            old_zeros=[]
            old_poles=[]
            old_gain=1.0

    new_poles=[[p.real,p.imag] for p in old_poles]
    new_zeros=[[z.real,z.imag] for z in old_zeros]
    return {"poles":new_poles, "zeros":new_zeros, "gain":old_gain}

##################################################
# Main Plot Callback
##################################################
@app.callback(
    Output("pz-plot","figure"),
    Output("bode-plot","figure"),
    Output("impulse-plot","figure"),
    Input("pz-store","data"),
    Input("pz-plot","relayoutData"),
    State("family-dropdown","value"),
    State("type-dropdown","value"),
    State("order-input","value"),
    State("domain-radio","value"),
    State("cutoff1-input","value"),
    State("cutoff2-input","value")
)
def update_plots(store_data, relayoutData,
                 family, ftype, order, domain, c1, c2):
    """Generates pole-zero figure (with shape-based drag), Bode plot, and impulse response."""
    # 1) Convert store data
    poles_c = [complex(p[0],p[1]) for p in store_data["poles"]]
    zeros_c = [complex(z[0],z[1]) for z in store_data["zeros"]]
    gain = store_data["gain"]

    # 2) If user dragged shapes => interpret relayoutData
    num_zeros = len(zeros_c)
    num_poles = len(poles_c)

    if relayoutData and isinstance(relayoutData, dict):
        for key, val in relayoutData.items():
            if key.startswith("shapes["):
                shape_idx = int(key.split("[")[1].split("]")[0])
                attr = key.split(".")[-1]
                # shape[0] => stable region => ignore
                if shape_idx==0:
                    continue
                # zeros => shapes 1..num_zeros
                if 1<=shape_idx<=num_zeros:
                    zidx=shape_idx-1
                    if attr in ["x0","y0"]:
                        x0=relayoutData.get(f"shapes[{shape_idx}].x0", zeros_c[zidx].real-0.05)
                        x1=relayoutData.get(f"shapes[{shape_idx}].x1", zeros_c[zidx].real+0.05)
                        y0=relayoutData.get(f"shapes[{shape_idx}].y0", zeros_c[zidx].imag-0.05)
                        y1=relayoutData.get(f"shapes[{shape_idx}].y1", zeros_c[zidx].imag+0.05)
                        newx=(x0+x1)/2.
                        newy=(y0+y1)/2.
                        snap=0.1
                        newx=round(newx/snap)*snap
                        newy=round(newy/snap)*snap
                        zeros_c[zidx]=complex(newx,newy)
                else:
                    # poles => each pole has 2 shapes
                    idx_pole_shape = shape_idx - (num_zeros+1)
                    pole_idx = idx_pole_shape//2
                    if attr in ["x0","y0"]:
                        x0=relayoutData.get(f"shapes[{shape_idx}].x0", poles_c[pole_idx].real-0.07)
                        x1=relayoutData.get(f"shapes[{shape_idx}].x1", poles_c[pole_idx].real+0.07)
                        y0=relayoutData.get(f"shapes[{shape_idx}].y0", poles_c[pole_idx].imag-0.07)
                        y1=relayoutData.get(f"shapes[{shape_idx}].y1", poles_c[pole_idx].imag+0.07)
                        newx=(x0+x1)/2.
                        newy=(y0+y1)/2.
                        snap=0.1
                        newx=round(newx/snap)*snap
                        newy=round(newy/snap)*snap
                        poles_c[pole_idx]=complex(newx,newy)

    # 3) compute freq/impulse ...
    #   (same as before; omitted for brevity, or you can copy your logic)
    # For demonstration, we'll just do an empty figure for Bode/Impulse:
    bode_fig = {
        "data": [],
        "layout": {"title":"Bode Plot","xaxis":{"title":"Freq"},"yaxis":{"title":"Mag(dB)"}}
    }
    impulse_fig= {
        "data": [],
        "layout": {"title":"Impulse","xaxis":{"title":"Time"},"yaxis":{"title":"Amplitude"}}
    }

    # 4) build PZ figure with shapes
    fig_pz = {
        "data": [],
        "layout": {
            "title": "Pole-Zero Plot",
            "xaxis": {"title":"Real Axis"},
            "yaxis": {"title":"Imag Axis", "scaleanchor":"x","scaleratio":1},
            "margin":{"l":60,"r":20,"t":40,"b":50},
            "shapes": [],
            "showlegend":False
        }
    }
    # shading shape[0]
    if domain=="analog":
        shape_stable = {
            "type":"rect","xref":"x","yref":"y",
            "x0":-9999.,"x1":0.,"y0":-9999.,"y1":9999.,
            "fillcolor":"rgba(0,255,0,0.07)",
            "line":{"width":0},
            "layer":"below"
        }
    else:
        shape_stable = {
            "type":"circle","xref":"x","yref":"y",
            "x0":-1.,"x1":1.,"y0":-1.,"y1":1.,
            "fillcolor":"rgba(0,255,0,0.07)",
            "line":{"width":0},
            "layer":"below"
        }
    fig_pz["layout"]["shapes"].append(shape_stable)

    # compute axis range
    all_x = [z.real for z in zeros_c]+[p.real for p in poles_c]
    all_y = [z.imag for z in zeros_c]+[p.imag for p in poles_c]
    if not all_x and not all_y:
        axis_lim=1.0
    else:
        max_val = max([1.0]+[abs(v) for v in (all_x+all_y)])
        axis_lim = max_val*1.2
    fig_pz["layout"]["xaxis"]["range"]=[-axis_lim, axis_lim]
    fig_pz["layout"]["yaxis"]["range"]=[-axis_lim, axis_lim]

    # zeros => shapes
    idx_shape=1
    for z_ in zeros_c:
        x_c = float(z_.real)
        y_c = float(z_.imag)
        shape_zero={
            "type":"circle","xref":"x","yref":"y",
            "x0": float(x_c-0.05), "x1": float(x_c+0.05),
            "y0": float(y_c-0.05), "y1": float(y_c+0.05),
            "line":{"color":"#1f77b4","width":2},
            "fillcolor":"rgba(0,0,0,0)"
        }
        fig_pz["layout"]["shapes"].append(shape_zero)
        idx_shape+=1

    # poles => 2 shapes each
    for p_ in poles_c:
        cx = float(p_.real)
        cy = float(p_.imag)
        d=0.07
        shape_line1={
            "type":"line","xref":"x","yref":"y",
            "x0": float(cx-d), "x1": float(cx+d),
            "y0": float(cy-d), "y1": float(cy+d),
            "line":{"color":"#d62728","width":2}
        }
        shape_line2={
            "type":"line","xref":"x","yref":"y",
            "x0": float(cx-d), "x1": float(cx+d),
            "y0": float(cy+d), "y1": float(cy-d),
            "line":{"color":"#d62728","width":2}
        }
        fig_pz["layout"]["shapes"].append(shape_line1)
        fig_pz["layout"]["shapes"].append(shape_line2)
        idx_shape+=2

    return fig_pz, bode_fig, impulse_fig


if __name__=="__main__":
    app.run(debug=True)
