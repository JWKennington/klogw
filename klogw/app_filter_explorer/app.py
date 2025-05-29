import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from scipy import signal

# A LIGO purple hue for highlighting
LIGO_PURPLE = "#593196"

##################################################
# MAIN APP INITIALIZATION
##################################################
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Signal Filter Visualizer (LIGO)"

# Inline CSS to handle layout, fonts, no-scroll approach on desktop, etc.
# If you prefer, put this in assets/style.css instead.
app.index_string = r"""
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <!-- Inline Lato font -->
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
    /* Force main row to fill leftover space on large screens, no scroll. */
    @media (min-width: 992px) {
        .main-content-row {
            height: calc(100vh - 210px) !important; /* Adjust offset as needed for your header/controls/footer height */
            overflow: hidden;
        }
    }
    /* Make each col fill height of the row */
    #left-col, #right-col {
        height: 100%;
    }
    /* In the right col, we have two graphs stacked, each 50%. */
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
    /* On medium+ screens, always show controls, hide toggle. */
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
    /* Ensure the dropdown menu can overflow and scroll if many options */
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
    <footer class="footer text-center">
        © Jim Kennington 2025
    </footer>
    {%config%}
    {%scripts%}
    {%renderer%}
</body>
</html>
"""

##################################################
# HEADER / NAVBAR
##################################################
LIGO_LOGO_URL = "https://dcc.ligo.org/public/0000/F0900035/002/ligo_logo.png"
header = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(src=LIGO_LOGO_URL, height="50px"), width="auto"
                        ),
                        dbc.Col(
                            dbc.NavbarBrand(
                                "Interactive Filter Visualization", className="ms-2"
                            )
                        ),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="#",
                style={"textDecoration": "none"},
            )
        ]
    ),
    color="white",
    dark=False,
    className="mb-0",
)

##################################################
# FILTER CONTROL COMPONENTS
##################################################
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

DEFAULT_FAMILY = "Butterworth"
DEFAULT_TYPE = "low"
DEFAULT_ORDER = 4
DEFAULT_DOMAIN = "analog"
DEFAULT_CUTOFF1 = 1.0
DEFAULT_CUTOFF2 = 2.0

domain_radio = dbc.RadioItems(
    id="domain-radio",
    className="btn-group",
    inputClassName="btn-check",
    labelClassName="btn btn-outline-primary",
    labelCheckedClassName="active",
    options=DOMAIN_OPTIONS,
    value=DEFAULT_DOMAIN,
)
domain_radio_group = html.Div(
    domain_radio, className="btn-group me-2", **{"role": "group"}
)

family_dd = dcc.Dropdown(
    id="family-dropdown",
    options=FILTER_FAMILIES,
    value=DEFAULT_FAMILY,
    clearable=False,
    style={"minWidth": "120px"},
)
type_dd = dcc.Dropdown(
    id="type-dropdown",
    options=FILTER_TYPES,
    value=DEFAULT_TYPE,
    clearable=False,
    style={"minWidth": "100px"},
)
order_input = dbc.Input(
    id="order-input",
    type="number",
    value=DEFAULT_ORDER,
    min=1,
    step=1,
    style={"width": "5ch"},
)

cut1_in = dbc.Input(
    id="cutoff1-input",
    type="number",
    value=DEFAULT_CUTOFF1,
    step=0.1,
    style={"width": "6ch"},
)
cut2_in = dbc.Input(
    id="cutoff2-input",
    type="number",
    value=DEFAULT_CUTOFF2,
    step=0.1,
    style={"width": "6ch"},
)
cut1_label = dbc.InputGroupText("Cutoff 1")
cut2_label = dbc.InputGroupText("Cutoff 2")
cut1_grp = dbc.InputGroup([cut1_label, cut1_in], id="cutoff1-group", className="me-2")
cut2_grp = dbc.InputGroup(
    [cut2_label, cut2_in], id="cutoff2-group", style={"display": "none"}
)

controls_row = dbc.Row(
    [
        dbc.Col(domain_radio_group, width="auto"),
        dbc.Col(family_dd, width="auto"),
        dbc.Col(type_dd, width="auto"),
        dbc.Col(order_input, width="auto"),
        dbc.Col(cut1_grp, width="auto"),
        dbc.Col(cut2_grp, width="auto"),
    ],
    align="center",
    className="g-2 flex-wrap",
)

controls_collapse = dbc.Collapse(controls_row, id="controls-collapse", is_open=False)
toggle_btn = dbc.Button(
    "Filter Controls",
    id="controls-toggle-btn",
    color="secondary",
    className="d-md-none mb-2",
)

##################################################
# POLE-ZERO BUTTONS
##################################################
pz_buttons = html.Div(
    [
        dbc.Button(
            "Add Pole",
            id="add-pole-btn",
            color="primary",
            outline=True,
            size="sm",
            className="me-2",
        ),
        dbc.Button(
            "Add Zero",
            id="add-zero-btn",
            color="primary",
            outline=True,
            size="sm",
            className="me-2",
        ),
        dbc.Button("Clear", id="clear-btn", color="secondary", size="sm"),
    ],
    className="mb-2",
)

##################################################
# GRAPHS
##################################################
pz_graph = dcc.Graph(
    id="pz-plot",
    config={
        "editable": True,
        "edits": {"shapePosition": True},
        "displayModeBar": False,
    },
    style={"width": "100%", "height": "100%"},
)
bode_graph = dcc.Graph(
    id="bode-plot",
    config={"displayModeBar": False},
    style={"width": "100%", "height": "50%"},
)
impulse_graph = dcc.Graph(
    id="impulse-plot",
    config={"displayModeBar": False},
    style={"width": "100%", "height": "50%"},
)

##################################################
# APP LAYOUT
##################################################
app.layout = html.Div(
    [
        header,
        dbc.Container([toggle_btn, controls_collapse], fluid=True, className="mt-2"),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [pz_graph, pz_buttons],
                        style={
                            "width": "100%",
                            "height": "100%",
                            "display": "flex",
                            "flexDirection": "column",
                        },
                    ),
                    md=6,
                    style={"height": "100%"},
                    id="left-col",
                ),
                dbc.Col(
                    html.Div(
                        [bode_graph, impulse_graph],
                        style={
                            "width": "100%",
                            "height": "100%",
                            "display": "flex",
                            "flexDirection": "column",
                        },
                    ),
                    md=6,
                    style={"height": "100%"},
                    id="right-col",
                ),
            ],
            className="gx-0 main-content-row flex-nowrap",
            style={"margin": 0},
        ),
        dcc.Store(id="pz-store", data={"poles": [], "zeros": [], "gain": 1.0}),
    ],
    style={"display": "flex", "flexDirection": "column", "minHeight": "100vh"},
)


##################################################
# TOGGLE CONTROLS ON MOBILE
##################################################
@app.callback(
    Output("controls-collapse", "is_open"),
    Input("controls-toggle-btn", "n_clicks"),
    State("controls-collapse", "is_open"),
)
def toggle_controls(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


##################################################
# SHOW/HIDE SECOND CUTOFF
##################################################
@app.callback(
    Output("cutoff2-group", "style"),
    Output("cutoff1-label", "children"),
    Output("cutoff2-label", "children"),
    Input("type-dropdown", "value"),
    Input("domain-radio", "value"),
)
def toggle_cutoff_2(ftype, domain):
    if ftype in ["bandpass", "bandstop"]:
        style2 = {}
        c1l = "Low Cutoff"
        c2l = "High Cutoff"
    else:
        style2 = {"display": "none"}
        c1l = "Cutoff Freq"
        c2l = "Cutoff 2"
    if domain == "analog":
        c1l += " (rad/s)"
        c2l += " (rad/s)"
    else:
        c1l += " (norm.)"
        c2l += " (norm.)"
    return style2, c1l, c2l


##################################################
# DESIGN FILTER HELPER
##################################################
def design_filter(family, btype, order, domain, c1, c2=None):
    analog = domain == "analog"
    if btype in ["bandpass", "bandstop"]:
        lo = min(c1, c2)
        hi = max(c1, c2)
        if analog:
            if lo <= 0:
                lo = 1e-6
        else:
            if lo <= 0:
                lo = 1e-6
            if hi >= 1:
                hi = 0.999999
        Wn = [lo, hi]
    else:
        Wn = c1
        if analog:
            if Wn <= 0:
                Wn = 1e-6
        else:
            if Wn >= 1:
                Wn = 0.999999
            if Wn <= 0:
                Wn = 1e-6
    try:
        if family == "Butterworth":
            z, p, k = signal.butter(order, Wn, btype=btype, analog=analog, output="zpk")
        elif family == "Chebyshev I":
            z, p, k = signal.cheby1(
                order, 1, Wn, btype=btype, analog=analog, output="zpk"
            )
        elif family == "Chebyshev II":
            z, p, k = signal.cheby2(
                order, 40, Wn, btype=btype, analog=analog, output="zpk"
            )
        elif family == "Elliptic":
            z, p, k = signal.ellip(
                order, 1, 40, Wn, btype=btype, analog=analog, output="zpk"
            )
        elif family == "Bessel":
            z, p, k = signal.bessel(order, Wn, btype=btype, analog=analog, output="zpk")
        else:
            z = np.array([])
            p = np.array([])
            k = 1.0
    except:
        z = np.array([])
        p = np.array([])
        k = 1.0
    zeros = [[float(zr.real), float(zr.imag)] for zr in z]
    poles = [[float(pr.real), float(pr.imag)] for pr in p]
    return zeros, poles, float(k)


##################################################
# UPDATE THE POLE-ZERO STORE
##################################################
@app.callback(
    Output("pz-store", "data"),
    Input("family-dropdown", "value"),
    Input("type-dropdown", "value"),
    Input("order-input", "value"),
    Input("domain-radio", "value"),
    Input("cutoff1-input", "value"),
    Input("cutoff2-input", "value"),
    Input("add-pole-btn", "n_clicks"),
    Input("add-zero-btn", "n_clicks"),
    Input("clear-btn", "n_clicks"),
    State("pz-store", "data"),
)
def update_store(
    family, ftype, order, domain, c1, c2, addp, addz, clear_btn, store_data
):
    ctx = callback_context
    trig_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    old_poles = [complex(p[0], p[1]) for p in store_data.get("poles", [])]
    old_zeros = [complex(z[0], z[1]) for z in store_data.get("zeros", [])]
    old_gain = store_data.get("gain", 1.0)

    # if filter param changed
    if trig_id in [
        "family-dropdown",
        "type-dropdown",
        "order-input",
        "domain-radio",
        "cutoff1-input",
        "cutoff2-input",
    ]:
        if family != "Custom":
            zz, pp, kk = design_filter(family, ftype, order, domain, c1, c2)
            old_zeros = [complex(z[0], z[1]) for z in zz]
            old_poles = [complex(p[0], p[1]) for p in pp]
            old_gain = kk
        else:
            # if switching domain while custom, reset
            if trig_id == "domain-radio":
                old_zeros = []
                old_poles = []
                old_gain = 1.0

    # add pole
    if trig_id == "add-pole-btn":
        if domain == "analog":
            idx = len(old_poles)
            newp = complex(-0.5 * (idx + 1), 0)
            old_poles.append(newp)
        else:
            idx = len(old_poles)
            val = 0.5 + 0.2 * idx
            if val > 0.9:
                val = 0.9
            newp = complex(val, 0)
            old_poles.append(newp)
    # add zero
    if trig_id == "add-zero-btn":
        if domain == "analog":
            idx = len(old_zeros)
            newz = complex(-0.5 * (idx + 1), 0)
            old_zeros.append(newz)
        else:
            idx = len(old_zeros)
            if idx == 0:
                old_zeros.append(complex(1, 0))
            elif idx == 1:
                old_zeros.append(complex(-1, 0))
            else:
                old_zeros.append(complex(0.5, 0))

    # clear
    if trig_id == "clear-btn":
        if family != "Custom":
            zz, pp, kk = design_filter(family, ftype, order, domain, c1, c2)
            old_zeros = [complex(z[0], z[1]) for z in zz]
            old_poles = [complex(p[0], p[1]) for p in pp]
            old_gain = kk
        else:
            old_zeros = []
            old_poles = []
            old_gain = 1.0

    new_poles = [[p.real, p.imag] for p in old_poles]
    new_zeros = [[z.real, z.imag] for z in old_zeros]
    return {"poles": new_poles, "zeros": new_zeros, "gain": old_gain}


##################################################
# SHAPE-BASED DRAG, BODE, IMPULSE PLOTS
##################################################
@app.callback(
    Output("pz-plot", "figure"),
    Output("bode-plot", "figure"),
    Output("impulse-plot", "figure"),
    Input("pz-store", "data"),
    Input("pz-plot", "relayoutData"),
    State("family-dropdown", "value"),
    State("type-dropdown", "value"),
    State("order-input", "value"),
    State("domain-radio", "value"),
    State("cutoff1-input", "value"),
    State("cutoff2-input", "value"),
)
def update_plots(store_data, relayoutData, family, ftype, order, domain, c1, c2):
    """Generates the pole-zero figure with shape-based drag,
    plus the Bode and impulse response plots."""
    # Convert store data
    poles_c = [complex(p[0], p[1]) for p in store_data["poles"]]
    zeros_c = [complex(z[0], z[1]) for z in store_data["zeros"]]
    gain = store_data.get("gain", 1.0)

    # parse shape drags
    # shape[0] => stability region shading
    # shape[1..num_zeros] => each zero
    # then each pole has 2 shapes
    num_zeros = len(zeros_c)
    num_poles = len(poles_c)

    if relayoutData and isinstance(relayoutData, dict):
        for key, val in relayoutData.items():
            if key.startswith("shapes["):
                idx = int(key.split("[")[1].split("]")[0])
                attr = key.split(".")[-1]
                # shape 0 => region shape => ignore
                if idx == 0:
                    continue
                # zero shapes => 1..num_zeros
                if 1 <= idx <= num_zeros:
                    zidx = idx - 1
                    if attr in ["x0", "y0"]:
                        x0 = relayoutData.get(
                            f"shapes[{idx}].x0", zeros_c[zidx].real - 0.05
                        )
                        x1 = relayoutData.get(
                            f"shapes[{idx}].x1", zeros_c[zidx].real + 0.05
                        )
                        y0 = relayoutData.get(
                            f"shapes[{idx}].y0", zeros_c[zidx].imag - 0.05
                        )
                        y1 = relayoutData.get(
                            f"shapes[{idx}].y1", zeros_c[zidx].imag + 0.05
                        )
                        newx = (x0 + x1) / 2.0
                        newy = (y0 + y1) / 2.0
                        snap = 0.1
                        newx = round(newx / snap) * snap
                        newy = round(newy / snap) * snap
                        zeros_c[zidx] = complex(newx, newy)
                else:
                    # pole shape
                    # offset = num_zeros +1 => first pole shape
                    idx_pole_shape = idx - (num_zeros + 1)
                    pole_idx = idx_pole_shape // 2
                    if attr in ["x0", "y0"]:
                        x0 = relayoutData.get(
                            f"shapes[{idx}].x0", poles_c[pole_idx].real - 0.07
                        )
                        x1 = relayoutData.get(
                            f"shapes[{idx}].x1", poles_c[pole_idx].real + 0.07
                        )
                        y0 = relayoutData.get(
                            f"shapes[{idx}].y0", poles_c[pole_idx].imag - 0.07
                        )
                        y1 = relayoutData.get(
                            f"shapes[{idx}].y1", poles_c[pole_idx].imag + 0.07
                        )
                        newx = (x0 + x1) / 2.0
                        newy = (y0 + y1) / 2.0
                        snap = 0.1
                        newx = round(newx / snap) * snap
                        newy = round(newy / snap) * snap
                        poles_c[pole_idx] = complex(newx, newy)

    # now recompute freq/impulse
    def zpk_freq_response(zeros, poles, k):
        analog = domain == "analog"
        if analog:
            # pick freq range
            if ftype in ["bandpass", "bandstop"]:
                lo = min(c1, c2)
                hi = max(c1, c2)
                fmin = max(1e-3, 0.1 * lo)
                fmax = max(fmin * 10, 10 * hi)
            else:
                fmin = max(1e-3, 0.1 * c1)
                fmax = max(fmin * 10, 10 * c1)
            w = np.logspace(np.log10(fmin), np.log10(fmax), 500)
            s = 1j * w
            num = np.ones_like(s, dtype=complex)
            den = np.ones_like(s, dtype=complex)
            for zc in zeros:
                num *= s - zc
            for pc in poles:
                den *= s - pc
            H = k * num / den
            return w, H
        else:
            worN = 800
            w = np.linspace(0, np.pi, worN)
            ejw = np.exp(1j * w)
            num = np.ones_like(ejw, dtype=complex)
            den = np.ones_like(ejw, dtype=complex)
            for zc in zeros:
                num *= ejw - zc
            for pc in poles:
                den *= ejw - pc
            H = k * num / den
            return w, H

    w, H = zpk_freq_response(zeros_c, poles_c, gain)
    mag = 20.0 * np.log10(np.abs(H) + 1e-12)
    phase = np.unwrap(np.angle(H))
    phase_deg = phase * 180.0 / np.pi

    # Bode figure
    bode_fig = {
        "data": [
            {
                "x": w.tolist(),
                "y": mag.tolist(),
                "mode": "lines",
                "name": "Mag(dB)",
                "marker": {"color": LIGO_PURPLE},
                "yaxis": "y1",
            },
            {
                "x": w.tolist(),
                "y": phase_deg.tolist(),
                "mode": "lines",
                "name": "Phase(deg)",
                "marker": {"color": "#ff7f0e"},
                "yaxis": "y2",
            },
        ],
        "layout": {
            "title": "Frequency Response (Bode Plot)",
            "xaxis": {
                "title": (
                    "Frequency (rad/s)"
                    if domain == "analog"
                    else "Frequency (rad/sample)"
                )
            },
            "yaxis": {"title": "Magnitude (dB)"},
            "yaxis2": {"title": "Phase (deg)", "overlaying": "y", "side": "right"},
            "margin": {"l": 60, "r": 60, "t": 50, "b": 50},
            "showlegend": False,
        },
    }
    if domain == "analog":
        bode_fig["layout"]["xaxis"]["type"] = "log"

    # Impulse response
    # do quick approach
    impulse_fig = {
        "data": [],
        "layout": {
            "title": "Impulse Response",
            "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
        },
    }
    analog = domain == "analog"
    if analog:
        # approximate continuous
        if len(poles_c) == 0 and len(zeros_c) == 0:
            # trivial
            t_ = [0, 1e-3]
            h_ = [gain, 0]
        else:
            # try a range from slowest pole
            neg_poles = [p for p in poles_c if p.real < 0]
            if neg_poles:
                slowest = max(
                    [-1.0 / (p.real) for p in neg_poles if p.real != 0], default=1
                )
            else:
                slowest = 1
            tmax = min(slowest * 5, 100)
            t = np.linspace(0, tmax, 500)
            b, a = signal.zpk2tf(zeros_c, poles_c, gain)
            try:
                tout, yout = signal.impulse((b, a), T=t)
                t_ = tout
                h_ = yout
            except:
                t_ = t
                h_ = np.zeros_like(t)
        impulse_fig["data"].append({"x": t_, "y": h_, "mode": "lines", "name": "h(t)"})
        impulse_fig["layout"]["xaxis"] = {"title": "Time (s)"}
        impulse_fig["layout"]["yaxis"] = {"title": "Amplitude"}
    else:
        # discrete
        b, a = signal.zpk2tf(zeros_c, poles_c, gain)
        if len(poles_c) > 0:
            max_mag = max(abs(p) for p in poles_c)
        else:
            max_mag = 0
        if max_mag < 1:
            N = 100
        else:
            N = 200
        imp = np.zeros(N)
        imp[0] = 1.0
        h_ = signal.lfilter(b, a, imp)
        n_ = np.arange(N)
        impulse_fig["data"].append(
            {"x": n_.tolist(), "y": h_.tolist(), "mode": "lines", "name": "h[n]"}
        )
        impulse_fig["layout"]["xaxis"] = {"title": "Samples (n)"}
        impulse_fig["layout"]["yaxis"] = {"title": "Amplitude"}

    # build pz figure
    fig_pz = {
        "data": [],
        "layout": {
            "title": "Pole-Zero Plot",
            "xaxis": {"title": "Real Axis", "zeroline": True, "zerolinecolor": "#aaa"},
            "yaxis": {
                "title": "Imag Axis",
                "zeroline": True,
                "zerolinecolor": "#aaa",
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
            "shapes": [],
            "showlegend": False,
        },
        "config": {
            "editable": True,
            "edits": {"shapePosition": True},
            "displayModeBar": False,
        },
    }
    all_x = [z.real for z in zeros_c] + [p.real for p in poles_c]
    all_y = [z.imag for z in zeros_c] + [p.imag for p in poles_c]
    if not all_x and not all_y:
        axis_lim = 1.0
    else:
        max_val = max([1.0] + [abs(v) for v in (all_x + all_y)])
        axis_lim = max_val * 1.2
    fig_pz["layout"]["xaxis"]["range"] = [-axis_lim, axis_lim]
    fig_pz["layout"]["yaxis"]["range"] = [-axis_lim, axis_lim]

    # shape[0] => stability shading
    if domain == "analog":
        shape_stable = {
            "type": "rect",
            "xref": "x",
            "yref": "y",
            "x0": -9999,
            "x1": 0,
            "y0": -9999,
            "y1": 9999,
            "fillcolor": "rgba(0,255,0,0.07)",
            "line": {"width": 0},
            "layer": "below",
        }
    else:
        shape_stable = {
            "type": "circle",
            "xref": "x",
            "yref": "y",
            "x0": -1,
            "x1": 1,
            "y0": -1,
            "y1": 1,
            "fillcolor": "rgba(0,255,0,0.07)",
            "line": {"width": 0},
            "layer": "below",
        }
    fig_pz["layout"]["shapes"].append(shape_stable)

    # zeros => next shapes
    shape_idx = 1
    for zc in zeros_c:
        x0 = zc.real - 0.05
        x1 = zc.real + 0.05
        y0 = zc.imag - 0.05
        y1 = zc.imag + 0.05
        shape_zero = {
            "type": "circle",
            "xref": "x",
            "yref": "y",
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "line": {"color": "#1f77b4", "width": 2},
            "fillcolor": "rgba(0,0,0,0)",
        }
        fig_pz["layout"]["shapes"].append(shape_zero)
        shape_idx += 1
    # poles => each pole => 2 shapes
    for pc in poles_c:
        cx = pc.real
        cy = pc.imag
        d = 0.07
        l1 = {
            "type": "line",
            "xref": "x",
            "yref": "y",
            "x0": cx - d,
            "x1": cx + d,
            "y0": cy - d,
            "y1": cy + d,
            "line": {"color": "#d62728", "width": 2},
        }
        l2 = {
            "type": "line",
            "xref": "x",
            "yref": "y",
            "x0": cx - d,
            "x1": cx + d,
            "y0": cy + d,
            "y1": cy - d,
            "line": {"color": "#d62728", "width": 2},
        }
        fig_pz["layout"]["shapes"].append(l1)
        fig_pz["layout"]["shapes"].append(l2)
        shape_idx += 2

    return fig_pz, bode_fig, impulse_fig


##################################################
# RUN
##################################################
if __name__ == "__main__":
    app.run(debug=True)
