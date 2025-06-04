import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from scipy import signal

##################################################
# Constants & Defaults
##################################################

LIGO_PURPLE    = "#593196"
LIGO_BLUE      = "#1f77b4"
LIGO_RED       = "#d62728"
LIGO_LOGO_URL  = "https://dcc.ligo.org/public/0000/F0900035/002/ligo_logo.png"

DEFAULT_FAMILY   = "Butterworth"
DEFAULT_TYPE     = "low"
DEFAULT_ORDER    = 4
DEFAULT_DOMAIN   = "analog"
DEFAULT_CUTOFF1  = 1.0
DEFAULT_CUTOFF2  = 2.0

FILTER_FAMILIES = [
    {"label":"Butterworth",        "value":"Butterworth"},
    {"label":"Chebyshev I",        "value":"Chebyshev I"},
    {"label":"Chebyshev II",       "value":"Chebyshev II"},
    {"label":"Elliptic",           "value":"Elliptic"},
    {"label":"Bessel",             "value":"Bessel"},
    {"label":"GW Inspiral Approx", "value":"GW"},
    {"label":"Custom (manual)",    "value":"Custom"},
]
FILTER_TYPES = [
    {"label":"Lowpass",  "value":"low"},
    {"label":"Highpass", "value":"high"},
    {"label":"Bandpass", "value":"bandpass"},
    {"label":"Bandstop", "value":"bandstop"}
]
DOMAIN_OPTIONS = [
    {"label":"Analog",  "value":"analog"},
    {"label":"Digital", "value":"digital"}
]

##################################################
# App Initialization
##################################################

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name":"viewport","content":"width=device-width, initial-scale=1"}],
)
app.title = "Signal Filter Visualization (LIGO)"

# Inline CSS & index_string
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
      font-family:'Lato',sans-serif; margin:0; padding:0; background:#fff;
    }
    .footer {
      text-align:center; padding:10px; border-top:1px solid #eaeaea;
      background:#f8f9fa; color:#666;
    }
    @media(min-width:992px){
      .main-content-row {
        height:calc(100vh - 230px)!important; overflow:hidden;
      }
    }
    #left-col,#right-col{ height:100%; }
    #right-col>div{ width:100%; height:100%; display:flex; flex-direction:column; }
    #right-col>div>div:first-child{ height:50%; }
    #right-col>div>div:last-child{  height:50%; }
    @media(min-width:768px){
      #controls-collapse.collapse{
        display:block!important; visibility:visible!important; height:auto!important;
      }
      #controls-toggle-btn{ display:none!important; }
    }
    .dropdown-menu{ max-height:250px; overflow-y:auto; z-index:2000; }
    .input-group .form-control {
      min-width: 6ch;
    }
    </style>
    {%renderer%}
</head>
<body>
    {%app_entry%}
    <footer class="footer">© Jim Kennington 2025</footer>
    {%config%}
    {%scripts%}
    {%renderer%}
</body>
</html>
"""

##################################################
# Layout: Header, Controls, Graphs
##################################################

header = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.Img(src=LIGO_LOGO_URL, height="50px")),
                dbc.Col(dbc.NavbarBrand("Interactive Filter Visualization", className="ms-2"))
            ], align="center", className="g-0"),
            href="#", style={"textDecoration":"none"}
        )
    ]),
    color="white", dark=False, className="mb-0"
)

# 1) Domain (Analog / Digital)
domain_radio = dbc.RadioItems(
    id="domain-radio", className="btn-group",
    inputClassName="btn-check", labelClassName="btn btn-outline-primary",
    labelCheckedClassName="active",
    options=DOMAIN_OPTIONS, value=DEFAULT_DOMAIN
)
domain_radio_group = html.Div(domain_radio, className="btn-group me-2", **{"role":"group"})

# 2) Filter family & type
family_dd = dcc.Dropdown(
    id="family-dropdown", options=FILTER_FAMILIES,
    value=DEFAULT_FAMILY, clearable=False,
    style={"minWidth":"140px"}
)
type_dd = dcc.Dropdown(
    id="type-dropdown", options=FILTER_TYPES,
    value=DEFAULT_TYPE, clearable=False,
    style={"minWidth":"100px"}
)

# 3) Order selector (wider + labeled)
order_group = dbc.InputGroup(
    [
        dbc.InputGroupText("Order"),
        dbc.Input(
            id="order-input", type="number",
            value=DEFAULT_ORDER, min=1, step=1,
            style={"width":"6ch"}
        )
    ],
    className="me-2",
)

# 4) Cutoff 1 & 2
cut1_label = dbc.InputGroupText(id="cutoff1-label", children="Cutoff 1")
cut1_in    = dbc.Input(
    id="cutoff1-input", type="number",
    value=DEFAULT_CUTOFF1, step=0.1,
    style={"width":"6ch"}
)
cut1_grp   = dbc.InputGroup([cut1_label, cut1_in], id="cutoff1-group", className="me-2")

cut2_label = dbc.InputGroupText(id="cutoff2-label", children="Cutoff 2")
cut2_in    = dbc.Input(
    id="cutoff2-input", type="number",
    value=DEFAULT_CUTOFF2, step=0.1,
    style={"width":"6ch"}
)
cut2_grp   = dbc.InputGroup([cut2_label, cut2_in], id="cutoff2-group", style={"display":"none"})

# Controls row (wraps on mobile)
controls_row = dbc.Row([
    dbc.Col(domain_radio_group,    width="auto"),
    dbc.Col(family_dd,            width="auto"),
    dbc.Col(type_dd,              width="auto"),
    dbc.Col(order_group,          width="auto"),
    dbc.Col(cut1_grp,             width="auto"),
    dbc.Col(cut2_grp,             width="auto"),
], align="center", className="g-2 flex-wrap")

controls_collapse = dbc.Collapse(controls_row, id="controls-collapse", is_open=False)
toggle_btn = dbc.Button("Filter Controls", id="controls-toggle-btn", color="secondary", className="d-md-none mb-2")

# Pole/Zero buttons + match/remove
pz_buttons = html.Div([
    dbc.Button("Add Pole",         id="add-pole-btn",    color="primary", outline=True, size="sm", className="me-2"),
    dbc.Button("Remove Last Pole", id="remove-pole-btn", color="danger",  outline=True, size="sm", className="me-2"),
    dbc.Button("Add Zero",         id="add-zero-btn",    color="primary", outline=True, size="sm", className="me-2"),
    dbc.Button("Remove Last Zero", id="remove-zero-btn", color="danger",  outline=True, size="sm", className="me-2"),
    dbc.Button("Match Conjugates", id="match-btn",       color="secondary",         size="sm", className="me-2"),
    dbc.Button("Clear All",        id="clear-btn",       color="secondary",         size="sm"),
], className="mb-2")

# Graphs: PZ remains editable (shapes only), but disables title/axis edits
pz_graph = dcc.Graph(
    id="pz-plot",
    style={"width":"100%","height":"100%"},
    config={
        "editable": True,
        "edits": {
            "shapePosition": True,
            "titleText": False,
            "axisTitleText": False,
            "annotationText": False,
            "annotationPosition": False
        },
        "displayModeBar": False
    }
)
bode_graph = dcc.Graph(
    id="bode-plot",
    style={"width":"100%","height":"50%"},
    config={"displayModeBar":False, "editable":False}
)
impulse_graph = dcc.Graph(
    id="impulse-plot",
    style={"width":"100%","height":"50%"},
    config={"displayModeBar":False, "editable":False}
)

app.layout = html.Div([
    header,
    dbc.Container([toggle_btn, controls_collapse], fluid=True, className="mt-2"),

    dbc.Row([
        dbc.Col(
            html.Div([pz_graph, pz_buttons],
                     style={"width":"100%","height":"100%","display":"flex","flexDirection":"column"}),
            md=6, style={"height":"100%"}, id="left-col"
        ),
        dbc.Col(
            html.Div([bode_graph, impulse_graph],
                     style={"width":"100%","height":"100%","display":"flex","flexDirection":"column"}),
            md=6, style={"height":"100%"}, id="right-col"
        )
    ], className="gx-0 main-content-row flex-nowrap", style={"margin":0}),

    # Stores: poles/zeros/gain, plus shape‐ID tracking
    dcc.Store(id="pz-store",    data={"poles":[],"zeros":[],"gain":1.0}),
    dcc.Store(id="shapes-meta", data=[]),
], style={"display":"flex","flexDirection":"column","minHeight":"100vh"})


##################################################
# JSON sanitize (1e-12 imaginary tolerance)
##################################################

def sanitize_json(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    if isinstance(obj, complex):
        if abs(obj.imag) < 1e-12:
            return float(obj.real)
        else:
            raise ValueError(f"Found truly complex numeric: {obj}")
    if isinstance(obj, (np.float64, np.float32, np.float16, np.int64, np.int32)):
        return float(obj)
    if hasattr(obj, "real") and hasattr(obj, "imag"):
        if abs(obj.imag) < 1e-12:
            return float(obj.real)
        else:
            raise ValueError(f"Found truly complex numeric: {obj}")
    if isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    return obj


##################################################
# Filter design (incl. “GW Inspiral Approx” logic)
##################################################

def design_filter(family, ftype, order, domain, c1, c2):
    import numpy as np
    from scipy import signal

    analog = (domain == "analog")

    # If “GW Inspiral Approx” is selected:
    if family == "GW":
        # Three complex‐conjugate analog pole pairs:
        #   Pair 1: ~30 Hz, Q=10
        #   Pair 2: ~60 Hz, Q=10
        #   Pair 3: ~120 Hz, Q=10
        Q = 10.0
        freqs = [30, 60, 120]  # Hz
        poles = []
        for f0 in freqs:
            ωn = 2 * np.pi * f0
            damping = ωn / (2 * Q)
            real_part = -damping
            imag_part = ωn * np.sqrt(1.0 - 1.0/(4*Q*Q))
            poles.append(complex(real_part,  imag_part))
            poles.append(complex(real_part, -imag_part))
        zeros = np.array([])
        k = 1.0

        if not analog:
            # Bilinear transform → digital poles
            p_dig = []
            for p in poles:
                # z = (1 + p/2)/(1 - p/2)
                zd = (1.0 + p/2.0) / (1.0 - p/2.0)
                p_dig.append(zd)
            poles = np.array(p_dig)
            zeros = np.array([])  # none to map
            # Gain stays ≈1

        return (
            [[float(zr.real), float(zr.imag)] for zr in zeros],
            [[float(pr.real), float(pr.imag)] for pr in poles],
            float(k)
        )

    # Otherwise proceed with the usual families:
    if ftype in ["bandpass", "bandstop"]:
        lo = min(c1, c2)
        hi = max(c1, c2)
        if analog:
            if lo <= 0: lo = 1e-6
        else:
            if lo <= 0: lo = 1e-6
            if hi >= 1: hi = 0.999999
            if lo >= hi:
                lo, hi = 0.2, 0.5
        Wn = [lo, hi]
    else:
        Wn = c1
        if analog:
            if Wn <= 0: Wn = 1e-6
        else:
            if Wn >= 1: Wn = 0.999999
            if Wn <= 0: Wn = 1e-6

    try:
        if family == "Butterworth":
            z, p, k = signal.butter(order, Wn, btype=ftype, analog=analog, output="zpk")
        elif family == "Chebyshev I":
            z, p, k = signal.cheby1(order, 1, Wn, btype=ftype, analog=analog, output="zpk")
        elif family == "Chebyshev II":
            z, p, k = signal.cheby2(order, 40, Wn, btype=ftype, analog=analog, output="zpk")
        elif family == "Elliptic":
            z, p, k = signal.ellip(order, 1, 40, Wn, btype=ftype, analog=analog, output="zpk")
        elif family == "Bessel":
            z, p, k = signal.bessel(order, Wn, btype=ftype, analog=analog, output="zpk")
        else:
            z, p, k = np.array([]), np.array([]), 1.0
    except Exception:
        z, p, k = np.array([]), np.array([]), 1.0

    zeros = [[float(zr.real), float(zr.imag)] for zr in z]
    poles = [[float(pr.real), float(pr.imag)] for pr in p]
    return zeros, poles, float(k)


##################################################
# Toggle filter controls (mobile/desktop)
##################################################

@app.callback(
    Output("controls-collapse", "is_open"),
    Input("controls-toggle-btn", "n_clicks"),
    State("controls-collapse",   "is_open")
)
def toggle_controls(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


##################################################
# Show/hide second cutoff input
##################################################

@app.callback(
    Output("cutoff2-group", "style"),
    Output("cutoff1-label",   "children"),
    Output("cutoff2-label",   "children"),
    Input("type-dropdown",    "value"),
    Input("domain-radio",     "value")
)
def toggle_cut2(ftype, domain):
    if ftype in ["bandpass", "bandstop"]:
        style = {}
        c1 = "Low Cutoff"
        c2 = "High Cutoff"
    else:
        style = {"display":"none"}
        c1 = "Cutoff Freq"
        c2 = "Cutoff 2"

    if domain == "analog":
        c1 += " (rad/s)"
        c2 += " (rad/s)"
    else:
        c1 += " (norm.)"
        c2 += " (norm.)"
    return style, c1, c2


##################################################
# Unified callback: update store + figures
##################################################

@app.callback(
    Output("pz-store",      "data"),
    Output("shapes-meta",   "data"),
    Output("pz-plot",       "figure"),
    Output("bode-plot",     "figure"),
    Output("impulse-plot",  "figure"),
    Input("family-dropdown",   "value"),
    Input("type-dropdown",     "value"),
    Input("order-input",       "value"),
    Input("domain-radio",      "value"),
    Input("cutoff1-input",     "value"),
    Input("cutoff2-input",     "value"),
    Input("add-pole-btn",      "n_clicks"),
    Input("remove-pole-btn",   "n_clicks"),
    Input("add-zero-btn",      "n_clicks"),
    Input("remove-zero-btn",   "n_clicks"),
    Input("match-btn",         "n_clicks"),
    Input("clear-btn",         "n_clicks"),
    Input("pz-plot",           "relayoutData"),
    State("pz-store",          "data"),
    State("shapes-meta",       "data")
)
def update_all(fam, ftype, order, domain, c1, c2,
               addp, remp, addz, remz, match, clear_btn,
               relayoutData, store_data, shapes_meta):
    ctx = callback_context
    trig_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # 1) Read existing store
    old_poles = [complex(p[0], p[1]) for p in store_data["poles"]]
    old_zeros = [complex(z[0], z[1]) for z in store_data["zeros"]]
    old_gain  = store_data["gain"]

    # 2) If “GW Inspiral Approx” is selected, design that directly:
    if fam == "GW":
        zlist, plist, k_new = design_filter("GW", ftype, order, domain, c1, c2)
        old_zeros = [complex(zv[0], zv[1]) for zv in zlist]
        old_poles = [complex(pv[0], pv[1]) for pv in plist]
        old_gain  = k_new

    # 3) If filter parameters changed (and not Custom/GW), re‐design those:
    if trig_id in [
        "family-dropdown", "type-dropdown", "order-input",
        "domain-radio", "cutoff1-input", "cutoff2-input"
    ]:
        if fam != "Custom" and fam != "GW":
            zlist, plist, k_new = design_filter(fam, ftype, order, domain, c1, c2)
            old_zeros = [complex(zv[0], zv[1]) for zv in zlist]
            old_poles = [complex(pv[0], pv[1]) for pv in plist]
            old_gain  = k_new
        elif fam == "Custom":
            if trig_id == "domain-radio":
                old_zeros = []; old_poles = []; old_gain = 1.0

    # 4) Add a pole
    if trig_id == "add-pole-btn":
        if domain == "analog":
            idx = len(old_poles)
            old_poles.append(complex(-0.5*(idx+1), 0))
        else:
            idx = len(old_poles)
            val = 0.5 + 0.2*idx
            if val > 0.9: val = 0.9
            old_poles.append(complex(val, 0))

    # 5) Remove last pole
    if trig_id == "remove-pole-btn":
        if old_poles:
            old_poles.pop()

    # 6) Add a zero
    if trig_id == "add-zero-btn":
        if domain == "analog":
            idx = len(old_zeros)
            old_zeros.append(complex(-0.5*(idx+1), 0))
        else:
            idx = len(old_zeros)
            if idx == 0:
                old_zeros.append(complex(1, 0))
            elif idx == 1:
                old_zeros.append(complex(-1, 0))
            else:
                old_zeros.append(complex(0.5, 0))

    # 7) Remove last zero
    if trig_id == "remove-zero-btn":
        if old_zeros:
            old_zeros.pop()

    # 8) Match conjugates
    if trig_id == "match-btn":
        new_poles = []
        seen = set()
        for p in old_poles:
            if abs(p.imag) < 1e-12:
                new_poles.append(p)
            else:
                conj_p = complex(p.real, -p.imag)
                keyp = (round(p.real,12), round(p.imag,12))
                keyc = (round(conj_p.real,12), round(conj_p.imag,12))
                if keyp in seen or keyc in seen:
                    continue
                new_poles.append(p)
                new_poles.append(conj_p)
                seen.add(keyp); seen.add(keyc)
        old_poles = new_poles

        new_zeros = []
        seenz = set()
        for z in old_zeros:
            if abs(z.imag) < 1e-12:
                new_zeros.append(z)
            else:
                conj_z = complex(z.real, -z.imag)
                keyz  = (round(z.real,12), round(z.imag,12))
                keycz = (round(conj_z.real,12), round(conj_z.imag,12))
                if keyz in seenz or keycz in seenz:
                    continue
                new_zeros.append(z)
                new_zeros.append(conj_z)
                seenz.add(keyz); seenz.add(keycz)
        old_zeros = new_zeros

    # 9) Clear
    if trig_id == "clear-btn":
        if fam != "Custom" and fam != "GW":
            zlist, plist, k_new = design_filter(fam, ftype, order, domain, c1, c2)
            old_zeros = [complex(zv[0], zv[1]) for zv in zlist]
            old_poles = [complex(pv[0], pv[1]) for pv in plist]
            old_gain  = k_new
        else:
            old_zeros = []; old_poles = []; old_gain = 1.0

    # 10) Build shapes_meta_new in the same order we append shapes below
    shapes_meta_new = []
    if domain == "analog":
        shapes_meta_new.append("stable-region")
    else:
        shapes_meta_new.append("stable-region-fill")
        shapes_meta_new.append("stable-region-border")

    for zidx in range(len(old_zeros)):
        shapes_meta_new.append(f"zero-{zidx}")
    for pidx in range(len(old_poles)):
        shapes_meta_new.append(f"pole-{pidx}")

    # 11) If user dragged shapes → parse relayoutData
    if trig_id == "pz-plot" and isinstance(relayoutData, dict):
        by_idx = {}
        for key, val in relayoutData.items():
            if not key.startswith("shapes["):
                continue
            idx_str = key.split("[")[1].split("]")[0]
            idx = int(idx_str)
            attr = key.split(".")[-1]
            by_idx.setdefault(idx, {})[attr] = val

        for idx, coords_dict in by_idx.items():
            if idx >= len(shapes_meta):
                continue
            shape_id = shapes_meta[idx]
            if shape_id.startswith("stable-region"):
                continue
            x0 = coords_dict.get("x0", 0.0)
            x1 = coords_dict.get("x1", 0.0)
            y0 = coords_dict.get("y0", 0.0)
            y1 = coords_dict.get("y1", 0.0)
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0

            if shape_id.startswith("zero-"):
                zidx = int(shape_id.split("-")[1])
                if 0 <= zidx < len(old_zeros):
                    old_zeros[zidx] = complex(cx, cy)
            elif shape_id.startswith("pole-"):
                pidx = int(shape_id.split("-")[1])
                if 0 <= pidx < len(old_poles):
                    old_poles[pidx] = complex(cx, cy)

    # 12) Recompute Bode & Impulse for filter-based
    def freqz_zpk(zlist, plist, k):
        analog = (domain == "analog")
        if analog:
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
            for zz in zlist:
                num *= (s - zz)
            for pp in plist:
                den *= (s - pp)
            H = k * num / den
            return w, H
        else:
            w = np.linspace(0, np.pi, 800)
            ejw = np.exp(1j * w)
            num = np.ones_like(ejw, dtype=complex)
            den = np.ones_like(ejw, dtype=complex)
            for zz in zlist:
                num *= (ejw - zz)
            for pp in plist:
                den *= (ejw - pp)
            H = k * num / den
            return w, H

    w, H = freqz_zpk(old_zeros, old_poles, old_gain)
    mag = 20. * np.log10(np.abs(H) + 1e-12)
    phase = np.unwrap(np.angle(H))
    phase_deg = phase * 180. / np.pi

    # Build Bode figure with explicit nested titles & axis labels
    bode_fig = {
        "data": [
            {
                "x": w.tolist(),
                "y": mag.tolist(),
                "mode": "lines",
                "name": "Magnitude (dB)",
                "marker": {"color": LIGO_PURPLE},
                "yaxis": "y1"
            },
            {
                "x": w.tolist(),
                "y": phase_deg.tolist(),
                "mode": "lines",
                "name": "Phase (deg)",
                "marker": {"color": LIGO_RED},
                "yaxis": "y2"
            }
        ],
        "layout": {
            "title": {"text": "Frequency Response (Bode Plot)", "x": 0.5},
            "margin": {"l": 60, "r": 60, "t": 60, "b": 50},
            "showlegend": False,
            "xaxis": {"title": {"text": "Frequency (rad/s)" if domain=="analog" else "Frequency (rad/sample)"}},
            "yaxis": {"title": {"text": "Magnitude (dB)"}},
            "yaxis2": {"title": {"text": "Phase (deg)"}, "overlaying": "y", "side": "right"}
        }
    }
    if domain == "analog":
        bode_fig["layout"]["xaxis"]["type"] = "log"

    # Build Impulse figure (nested titles/axis labels)
    impulse_fig = {
        "data": [],
        "layout": {
            "title": {"text": "Impulse Response", "x": 0.5},
            "margin": {"l": 60, "r": 20, "t": 60, "b": 50},
            "xaxis": {},
            "yaxis": {}
        }
    }
    analog = (domain == "analog")
    if analog:
        if not old_poles and not old_zeros:
            t_ = [0, 1e-3]
            h_ = np.array([old_gain, 0], dtype=complex)
        else:
            neg_p = [p for p in old_poles if p.real < 0]
            if neg_p:
                slowest = max([-1.0 / p.real for p in neg_p if p.real < 0], default=1)
            else:
                slowest = 1
            tmax = min(slowest * 5, 100)
            t = np.linspace(0, tmax, 500)
            b, a = signal.zpk2tf(old_zeros, old_poles, old_gain)
            try:
                tout, yout = signal.impulse((b, a), T=t)
                h_ = yout.astype(complex)
                t_ = tout
            except Exception:
                t_ = t
                h_ = np.zeros_like(t, dtype=complex)

        # Plot real & imag
        impulse_fig["data"].append({
            "x": t_,
            "y": np.real(h_).tolist(),
            "mode": "lines",
            "name": "Re{h(t)}",
            "marker": {"color": LIGO_PURPLE}
        })
        impulse_fig["data"].append({
            "x": t_,
            "y": np.imag(h_).tolist(),
            "mode": "lines",
            "name": "Im{h(t)}",
            "marker": {"color": LIGO_RED}
        })
        impulse_fig["layout"]["xaxis"] = {"title": {"text": "Time (s)"}}
        impulse_fig["layout"]["yaxis"] = {"title": {"text": "Amplitude"}}

    else:
        b, a = signal.zpk2tf(old_zeros, old_poles, old_gain)
        if old_poles:
            max_mag = max(abs(pp) for pp in old_poles)
        else:
            max_mag = 0
        N = 100 if max_mag < 1 else 200
        imp = np.zeros(N)
        imp[0] = 1.0
        h_ = signal.lfilter(b, a, imp).astype(complex)
        n_ = np.arange(N)

        impulse_fig["data"].append({
            "x": n_.tolist(),
            "y": np.real(h_).tolist(),
            "mode": "lines",
            "name": "Re{h[n]}",
            "marker": {"color": LIGO_PURPLE}
        })
        impulse_fig["data"].append({
            "x": n_.tolist(),
            "y": np.imag(h_).tolist(),
            "mode": "lines",
            "name": "Im{h[n]}",
            "marker": {"color": LIGO_RED}
        })
        impulse_fig["layout"]["xaxis"] = {"title": {"text": "Samples (n)"}}
        impulse_fig["layout"]["yaxis"] = {"title": {"text": "Amplitude"}}

    # Build Pole‐Zero figure with unit‐circle/left‐half shading
    fig_pz = {
        "data": [],
        "layout": {
            "title": {"text": "Pole‐Zero Plot", "x": 0.5},
            "uirevision": "pz-uirev",
            "xaxis": {
                "title": {"text": "Real Axis"},
                "zeroline": True, "zerolinecolor": "#aaa"
            },
            "yaxis": {
                "title": {"text": "Imag Axis"},
                "scaleanchor": "x", "scaleratio": 1,
                "zeroline": True, "zerolinecolor": "#aaa"
            },
            "margin": {"l":60,"r":20,"t":60,"b":50},
            "shapes": [],
            "showlegend": False
        }
    }

    if domain == "analog":
        shape_stable = {
            "type": "rect", "xref": "x", "yref": "y",
            "x0": -9999, "x1": 0, "y0": -9999, "y1": 9999,
            "fillcolor": "rgba(0,255,0,0.07)",
            "line": {"width": 0},
            "layer": "below",
            "editable": False
        }
        fig_pz["layout"]["shapes"].append(shape_stable)
    else:
        # Digital: unit‐circle fill + dashed border
        shape_circle_fill = {
            "type":"circle","xref":"x","yref":"y",
            "x0":-1.0,"x1":1.0,"y0":-1.0,"y1":1.0,
            "fillcolor":"rgba(0,255,0,0.07)",
            "line":{"width":0},
            "layer":"below",
            "editable":False
        }
        shape_circle_border = {
            "type":"circle","xref":"x","yref":"y",
            "x0":-1.0,"x1":1.0,"y0":-1.0,"y1":1.0,
            "line":{"color":"black","width":1,"dash":"dash"},
            "fillcolor":"rgba(0,0,0,0)",
            "layer":"below",
            "editable":False
        }
        fig_pz["layout"]["shapes"].append(shape_circle_fill)
        fig_pz["layout"]["shapes"].append(shape_circle_border)

    # Draw zeros
    for zidx, z_ in enumerate(old_zeros):
        zx = float(z_.real)
        zy = float(z_.imag)
        shape_zero = {
            "type":"circle","xref":"x","yref":"y",
            "x0":zx-0.05,"x1":zx+0.05,
            "y0":zy-0.05,"y1":zy+0.05,
            "line":{"color":LIGO_BLUE,"width":2},
            "fillcolor":"rgba(0,0,0,0)"
        }
        fig_pz["layout"]["shapes"].append(shape_zero)

    # Draw poles as small squares
    for pidx, p_ in enumerate(old_poles):
        px = float(p_.real)
        py = float(p_.imag)
        half = 0.07
        shape_pole = {
            "type":"rect","xref":"x","yref":"y",
            "x0":px-half,"x1":px+half,
            "y0":py-half,"y1":py+half,
            "line":{"color":LIGO_RED,"width":2},
            "fillcolor":"rgba(0,0,0,0)"
        }
        fig_pz["layout"]["shapes"].append(shape_pole)

    # Adjust axis limits so that (0,0) is centered
    all_x = [z.real for z in old_zeros] + [p.real for p in old_poles]
    all_y = [z.imag for z in old_zeros] + [p.imag for p in old_poles]
    if not all_x and not all_y:
        ax_lim = 1.0
    else:
        maxi = max([1.0] + [abs(v) for v in (all_x + all_y)])
        ax_lim = maxi * 1.2
    fig_pz["layout"]["xaxis"]["range"] = [-ax_lim, ax_lim]
    fig_pz["layout"]["yaxis"]["range"] = [-ax_lim, ax_lim]

    # Build updated stores
    new_store = {
        "poles": [[p.real, p.imag] for p in old_poles],
        "zeros": [[z.real, z.imag] for z in old_zeros],
        "gain": old_gain
    }

    return (
        new_store,
        shapes_meta_new,
        sanitize_json(fig_pz),
        sanitize_json(bode_fig),
        sanitize_json(impulse_fig)
    )


if __name__ == "__main__":
    app.run(debug=True)
