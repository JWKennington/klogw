import numpy as np
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from scipy import signal

# Initialize Dash app with Bootstrap (Flatly theme)
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Signal Filter Visualizer"  # title for the browser tab

# LIGO logo URL (public link)
LIGO_LOGO_URL = "https://dcc.ligo.org/public/0000/F0900035/002/ligo_logo.png"

# Define filter family options
FILTER_FAMILIES = [
    {"label": "Butterworth", "value": "Butterworth"},
    {"label": "Chebyshev I", "value": "Chebyshev I"},
    {"label": "Chebyshev II", "value": "Chebyshev II"},
    {"label": "Elliptic", "value": "Elliptic"},
    {"label": "Bessel", "value": "Bessel"},
    {"label": "Custom (manual)", "value": "Custom"}
]
FILTER_TYPES = [
    {"label": "Lowpass", "value": "low"},
    {"label": "Highpass", "value": "high"},
    {"label": "Bandpass", "value": "bandpass"},
    {"label": "Bandstop", "value": "bandstop"}
]
DOMAIN_OPTIONS = [
    {"label": "Analog", "value": "analog"},
    {"label": "Digital", "value": "digital"}
]

# Default initial values
DEFAULT_FAMILY = "Butterworth"
DEFAULT_TYPE = "low"
DEFAULT_ORDER = 4
DEFAULT_DOMAIN = "analog"
DEFAULT_CUTOFF1 = 1.0   # analog 1 rad/s
DEFAULT_CUTOFF2 = 2.0   # (only used for band types)

# Build header with LIGO logo
header = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.Img(src=LIGO_LOGO_URL, height="50px")),
                dbc.Col(dbc.NavbarBrand("Interactive Filter Visualization", className="ms-2"))
            ], align="center", className="g-0"),
            href="#", style={"textDecoration": "none"}
        ),
    ]),
    color="light",  # or "white"
    dark=False,
    className="mb-2"
)

# Filter controls components
domain_radio = dbc.RadioItems(
    id="domain-radio", className="btn-group", inputClassName="btn-check",
    labelClassName="btn btn-outline-primary", labelCheckedClassName="active",
    options=DOMAIN_OPTIONS, value=DEFAULT_DOMAIN
)
domain_radio_group = html.Div(domain_radio, className="btn-group me-2", **{"role": "group"})

family_dropdown = dcc.Dropdown(
    id="family-dropdown", options=FILTER_FAMILIES, value=DEFAULT_FAMILY,
    clearable=False, style={"minWidth": "150px"}
)
type_dropdown = dcc.Dropdown(
    id="type-dropdown", options=FILTER_TYPES, value=DEFAULT_TYPE,
    clearable=False, style={"minWidth": "120px"}
)
order_input = dbc.Input(
    id="order-input", type="number", value=DEFAULT_ORDER, min=1, max=20, step=1,
    style={"width": "6ch"}  # small width for 2-digit numbers
)
# Cutoff frequency inputs (two inputs for bandpass/bandstop)
cutoff1_input = dbc.Input(id="cutoff1-input", type="number", value=DEFAULT_CUTOFF1, step=0.1)
cutoff2_input = dbc.Input(id="cutoff2-input", type="number", value=DEFAULT_CUTOFF2, step=0.1)
# Input group labels for cutoff frequencies
cutoff1_label = dbc.InputGroupText(id="cutoff1-label", children="Cutoff Frequency")
cutoff2_label = dbc.InputGroupText(id="cutoff2-label", children="Cutoff 2")
cutoff1_group = dbc.InputGroup([cutoff1_label, cutoff1_input], id="cutoff1-group", className="me-2", style={"width": "200px"})
cutoff2_group = dbc.InputGroup([cutoff2_label, cutoff2_input], id="cutoff2-group", className="me-2", style={"width": "200px", "display": "none"})

# Assemble the controls row (will be placed inside a Collapse for responsiveness)
controls_row = dbc.Row([
    dbc.Col(domain_radio_group, width="auto"),
    dbc.Col(html.Div(family_dropdown, style={"minWidth": "150px"}), width="auto", className="me-2"),
    dbc.Col(html.Div(type_dropdown, style={"minWidth": "130px"}), width="auto", className="me-2"),
    dbc.Col(html.Div(order_input, style={"maxWidth": "4rem"}), width="auto", className="me-2"),
    dbc.Col(cutoff1_group, width="auto"),
    dbc.Col(cutoff2_group, width="auto")
], align="center", className="flex-wrap")

# Pole-zero control buttons (Add Pole, Add Zero, Clear)
pz_buttons = html.Div([
    dbc.Button("Add Pole", id="add-pole-btn", color="primary", outline=True, size="sm", className="me-2"),
    dbc.Button("Add Zero", id="add-zero-btn", color="primary", outline=True, size="sm", className="me-2"),
    dbc.Button("Clear", id="clear-btn", color="secondary", size="sm")
], className="mb-2")

# Graphs placeholders
pz_graph = dcc.Graph(id="pz-plot", config={"displayModeBar": False})
bode_graph = dcc.Graph(id="bode-plot", config={"displayModeBar": False})
impulse_graph = dcc.Graph(id="impulse-plot", config={"displayModeBar": False})

# Layout assembly
app.layout = html.Div([
    header,
    # Toggle button (visible on mobile only) and collapsible controls
    html.Div([
        dbc.Button("Filter Controls", id="controls-toggle-btn", color="secondary", className="d-md-none mb-2"),
        dbc.Collapse(controls_row, id="controls-collapse", is_open=False)
    ], className="container mb-3"),
    # Main content row: left (PZ plot + buttons) and right (Bode + impulse)
    dbc.Container([
        dbc.Row([
            dbc.Col(
                [pz_graph, pz_buttons], width=6
            ),
            dbc.Col(
                [bode_graph, impulse_graph], width=6
            )
        ], align="start")
    ], fluid=True, className="mb-4 flex-grow-1"),
    html.Footer("© Jim Kennington 2025", className="footer")
], style={"display": "flex", "flexDirection": "column", "minHeight": "100vh"})

# Callback to toggle the controls collapse on mobile
@app.callback(
    Output("controls-collapse", "is_open"),
    Input("controls-toggle-btn", "n_clicks"),
    State("controls-collapse", "is_open")
)
def toggle_controls(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Callback to show/hide second cutoff input and update labels based on filter type and domain
@app.callback(
    Output("cutoff2-group", "style"),
    Output("cutoff1-label", "children"),
    Output("cutoff2-label", "children"),
    Input("type-dropdown", "value"),
    Input("domain-radio", "value")
)
def update_cutoff_inputs(filter_type, domain):
    # Determine if second cutoff should be visible
    style2 = {"width": "200px", "display": "none"}
    label1 = "Cutoff Frequency"
    label2 = "Cutoff 2"
    if filter_type in ["bandpass", "bandstop"]:
        # Show second cutoff input for band filters
        style2 = {"width": "200px", "display": "inline-flex"}
        label1 = "Low Cutoff"
        label2 = "High Cutoff"
    # Append units depending on domain
    if domain == "analog":
        label1 += " (rad/s)"
        label2 += " (rad/s)"
    else:
        label1 += " (norm.)"
        label2 += " (norm.)"
    return style2, label1, label2

# Data store for current poles and zeros (complex values stored as [Re, Im] pairs) and gain
# We initialize it with the default filter's poles and zeros computed below in a separate callback.
app.layout.children.insert(2, dcc.Store(id="pz-store", data={"poles": [], "zeros": [], "gain": 1.0}))

# Helper function: design filter using SciPy and return zeros, poles, gain
def design_filter(family, btype, order, domain, cutoff1, cutoff2=None):
    analog = (domain == "analog")
    # Prepare cutoff frequencies for SciPy
    if btype in ["bandpass", "bandstop"]:
        # Ensure cutoff1 < cutoff2
        low = min(cutoff1, cutoff2) if cutoff2 is not None else cutoff1
        high = max(cutoff1, cutoff2) if cutoff2 is not None else cutoff1
        # Enforce valid ranges
        if analog:
            if low <= 0: low = 1e-6
        else:
            if low <= 0: low = 1e-6
            if high >= 1: high = 0.999
        Wn = [low, high]
    else:
        Wn = cutoff1
        if analog:
            if Wn <= 0: Wn = 1e-6
        else:
            if Wn >= 1: Wn = 0.999
            if Wn <= 0: Wn = 1e-6
    # Design filter using SciPy
    try:
        if family == "Butterworth":
            z, p, k = signal.butter(order, Wn, btype=btype, analog=analog, output='zpk')
        elif family == "Chebyshev I":
            z, p, k = signal.cheby1(order, 1, Wn, btype=btype, analog=analog, output='zpk')  # 1 dB ripple
        elif family == "Chebyshev II":
            z, p, k = signal.cheby2(order, 40, Wn, btype=btype, analog=analog, output='zpk')  # 40 dB stopband
        elif family == "Elliptic":
            z, p, k = signal.ellip(order, 1, 40, Wn, btype=btype, analog=analog, output='zpk')  # 1 dB, 40 dB
        elif family == "Bessel":
            z, p, k = signal.bessel(order, Wn, btype=btype, analog=analog, output='zpk')
        else:
            # Custom: no design (we'll rely on stored values), but ensure k=1
            z, p, k = np.array([]), np.array([]), 1.0
    except Exception as e:
        # In case of any design error, return empty (should not happen for valid inputs)
        z, p, k = np.array([]), np.array([]), 1.0
    # Convert to standard Python lists and round tiny imaginary parts for cleanliness
    poles = []
    zeros = []
    for pole in p:
        pole = complex(pole)
        if abs(pole.real) < 1e-12: pole = complex(0, pole.imag)
        if abs(pole.imag) < 1e-12: pole = complex(pole.real, 0)
        poles.append([pole.real, pole.imag])
    for zero in z:
        zero = complex(zero)
        if abs(zero.real) < 1e-12: zero = complex(0, zero.imag)
        if abs(zero.imag) < 1e-12: zero = complex(zero.real, 0)
        zeros.append([zero.real, zero.imag])
    return zeros, poles, float(k)

# Main callback: update filter design and plots whenever controls or add/clear actions change
@app.callback(
    Output("pz-store", "data"),
    Output("pz-plot", "figure"),
    Output("bode-plot", "figure"),
    Output("impulse-plot", "figure"),
    Input("family-dropdown", "value"),
    Input("type-dropdown", "value"),
    Input("order-input", "value"),
    Input("domain-radio", "value"),
    Input("cutoff1-input", "value"),
    Input("cutoff2-input", "value"),
    Input("add-pole-btn", "n_clicks"),
    Input("add-zero-btn", "n_clicks"),
    Input("clear-btn", "n_clicks"),
    State("pz-store", "data")
)
def update_filter(family, f_type, order, domain, cutoff1, cutoff2,
                  add_pole_clicks, add_zero_clicks, clear_clicks, store):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Copy current store data for updates
    current_poles = [complex(x[0], x[1]) for x in store.get("poles", [])]
    current_zeros = [complex(x[0], x[1]) for x in store.get("zeros", [])]
    current_k = store.get("gain", 1.0)

    # Flags for actions
    add_pole = (triggered_id == "add-pole-btn")
    add_zero = (triggered_id == "add-zero-btn")
    clear = (triggered_id == "clear-btn")
    family_changed = (triggered_id == "family-dropdown")
    type_changed = (triggered_id == "type-dropdown")
    order_changed = (triggered_id == "order-input")
    domain_changed = (triggered_id == "domain-radio")
    cutoff_changed = (triggered_id in ["cutoff1-input", "cutoff2-input"])

    # If any filter parameter (family, type, order, domain, cutoff) changed:
    if family_changed or type_changed or order_changed or domain_changed or cutoff_changed:
        if family != "Custom":
            # For standard families, redesign filter from scratch
            z_list, p_list, k_val = design_filter(family, f_type, order, domain, cutoff1, cutoff2)
            current_zeros = [complex(z[0], z[1]) for z in z_list]
            current_poles = [complex(p[0], p[1]) for p in p_list]
            current_k = k_val
        else:
            # If switched to Custom family:
            if family_changed:
                # If coming from a previous standard filter, keep its current poles/zeros as starting point (already in store).
                # Otherwise (already in custom and user re-selected custom), no change.
                pass
            # If domain toggled while in Custom: clear poles/zeros because cannot carry analog <-> digital directly
            if domain_changed:
                current_poles = []
                current_zeros = []
                current_k = 1.0
            # Ignore other changes (type/order in custom mode have no effect).
        # Reset any add counters (no explicit counter used, but any previous additions remain in current_poles/zeros).
    # Handle add/remove actions
    if add_pole:
        # Add a new pole at default position depending on domain
        if domain == "analog":
            # Negative real axis, spaced by -0.5 * (count+1)
            new_index = len(current_poles) + 1
            new_pole = -0.5 * new_index
        else:  # digital
            # Positive real inside unit circle (0.5, 0.7, 0.9 for first few, capped at 0.9)
            new_index = len(current_poles)
            # Sequence: 0.5, 0.7, 0.9, 0.9...
            new_pole_val = 0.5 + 0.2 * new_index
            if new_pole_val > 0.9:
                new_pole_val = 0.9
            new_pole = new_pole_val + 0j
        current_poles.append(new_pole)
    if add_zero:
        if domain == "analog":
            # Place analog zeros on negative real axis for simplicity
            new_index = len(current_zeros) + 1
            new_zero = -0.5 * new_index
        else:  # digital
            # First zero at z=1, second at z=-1, subsequent zeros at 0.5
            if len(current_zeros) == 0:
                new_zero = 1 + 0j  # zero at z=1 (cancels DC)
            elif len(current_zeros) == 1:
                new_zero = -1 + 0j  # zero at z=-1 (cancels Nyquist)
            else:
                new_zero = 0.5 + 0j  # additional zeros inside unit circle
        current_zeros.append(new_zero)
    if clear:
        if family != "Custom":
            # If a standard filter, reset to the designed base filter (recompute)
            z_list, p_list, k_val = design_filter(family, f_type, order, domain, cutoff1, cutoff2)
            current_zeros = [complex(z[0], z[1]) for z in z_list]
            current_poles = [complex(p[0], p[1]) for p in p_list]
            current_k = k_val
        else:
            # If custom, clear all
            current_zeros = []
            current_poles = []
            current_k = 1.0

    # Prepare the output store data (convert complex to [Re, Im])
    poles_out = [[c.real, c.imag] for c in current_poles]
    zeros_out = [[c.real, c.imag] for c in current_zeros]
    store_data = {"poles": poles_out, "zeros": zeros_out, "gain": current_k}

    # Generate Pole-Zero Plot figure
    fig_pz = {
        "data": [],
        "layout": {}
    }
    # If digital, include unit circle shape
    show_unit_circle = (domain == "digital")
    # Determine axis range to include all points and unit circle if needed
    all_re = [c.real for c in (current_poles + current_zeros) if c != 0]
    all_im = [c.imag for c in (current_poles + current_zeros) if c != 0]
    max_val = 1.0
    if all_re or all_im:
        max_re = max([abs(x) for x in all_re]) if all_re else 0
        max_im = max([abs(y) for y in all_im]) if all_im else 0
        max_val = max(1.0 if show_unit_circle else 0.0, max_re, max_im)
    # Add some margin
    axis_limit = max_val * 1.1 if max_val > 0 else 1.0
    # Plot poles and zeros
    if current_poles:
        pole_x = [c.real for c in current_poles]
        pole_y = [c.imag for c in current_poles]
        fig_pz["data"].append({
            "x": pole_x, "y": pole_y,
            "mode": "markers", "name": "Poles",
            "marker": {"symbol": "x-thin", "size": 10, "line": {"width": 2}, "color": "#d62728"}
        })
    if current_zeros:
        zero_x = [c.real for c in current_zeros]
        zero_y = [c.imag for c in current_zeros]
        fig_pz["data"].append({
            "x": zero_x, "y": zero_y,
            "mode": "markers", "name": "Zeros",
            "marker": {"symbol": "circle", "size": 10, "line": {"width": 2}, "color": "#1f77b4"}
        })
    # Axes and layout for PZ plot
    fig_pz["layout"] = {
        "title": "Pole-Zero Plot",
        "xaxis": {"title": "Real Axis", "zeroline": True, "zerolinecolor": "#888"},
        "yaxis": {"title": "Imag Axis", "zeroline": True, "zerolinecolor": "#888"},
        "showlegend": False,
        "margin": {"l": 40, "r": 10, "t": 40, "b": 40},
        "xaxis_range": [-axis_limit, axis_limit],
        "yaxis_range": [-axis_limit, axis_limit],
        "yaxis_scaleanchor": "x",  # lock aspect ratio
        "yaxis_scaleratio": 1
    }
    if show_unit_circle:
        # Draw unit circle (as a scatter or shape)
        theta = np.linspace(0, 2*np.pi, 360)
        fig_pz["data"].append({
            "x": np.cos(theta), "y": np.sin(theta),
            "mode": "lines", "name": "Unit Circle", "line": {"color": "gray", "dash": "dot"}
        })

    # Compute frequency response for Bode plot
    if domain == "analog":
        # Analog frequency response (log scale)
        # Determine frequency range based on cutoff(s)
        if f_type in ["bandpass", "bandstop"]:
            low_cut = min(cutoff1, cutoff2)
            high_cut = max(cutoff1, cutoff2)
            f_min = 0.1 * (low_cut if low_cut > 0 else 1)
            f_max = 10 * (high_cut if high_cut > 0 else (low_cut if low_cut > 0 else 1))
        else:
            f_min = 0.1 * cutoff1 if cutoff1 > 0 else 0.1
            f_max = 10 * cutoff1 if cutoff1 > 0 else 10
        # Ensure range makes sense
        if f_min <= 0:
            f_min = 0.1
        if f_max <= f_min:
            f_max = f_min * 10
        # Generate logarithmic frequency array
        w = np.logspace(np.log10(f_min), np.log10(f_max), 500)
        # Use SciPy freqs to get analog frequency response
        if current_poles or current_zeros:
            # Use zpk form if possible
            w, h = signal.freqs_zpk(current_zeros, current_poles, current_k, worN=w)
        else:
            # If no poles (trivial H(s)=k), just construct response
            h = np.full_like(w, current_k, dtype=complex)
        freq_axis = w  # rad/s
        x_label = "Frequency (rad/s)"
    else:
        # Digital frequency response (0 to pi)
        worN = 800
        if current_poles or current_zeros:
            w, h = signal.freqz_zpk(current_zeros, current_poles, current_k, worN=worN)
        else:
            # No poles (FIR or empty) -> use zpk2tf to get polynomial and then freqz for accuracy
            b, a = signal.zpk2tf(current_zeros, current_poles, current_k)
            w, h = signal.freqz(b, a, worN=worN)
        freq_axis = w  # in rad/sample (0 to pi)
        x_label = "Frequency (rad/sample)"
    # Compute magnitude (dB) and phase (deg), unwrap phase for smooth plot
    mag = 20 * np.log10(np.abs(h) + 1e-12)
    phase = np.unwrap(np.angle(h))
    phase_deg = phase * 180.0 / np.pi

    # Bode plot figure with two subplots (magnitude and phase)
    fig_bode = {
        "data": [
            {"x": freq_axis, "y": mag, "name": "Magnitude", "marker": {"color": "#1f77b4"}, "mode": "lines", "yaxis": "y1"},
            {"x": freq_axis, "y": phase_deg, "name": "Phase", "marker": {"color": "#ff7f0e"}, "mode": "lines", "yaxis": "y2"}
        ],
        "layout": {
            "title": "Frequency Response (Bode Plot)",
            "margin": {"l": 60, "r": 60, "t": 40, "b": 50},
            "xaxis": {"title": x_label, "domain": [0.0, 1.0]},
            "yaxis": {"title": "Magnitude (dB)", "titlefont": {"color": "#1f77b4"}, "tickfont": {"color": "#1f77b4"}},
            "yaxis2": {
                "title": "Phase (degrees)", "overlaying": "y", "side": "right",
                "titlefont": {"color": "#ff7f0e"}, "tickfont": {"color": "#ff7f0e"}
            },
            "showlegend": False
        }
    }
    # Use log scale for analog frequency axis
    if domain == "analog":
        fig_bode["layout"]["xaxis"]["type"] = "log"

    # Impulse response computation
    fig_imp = {
        "data": [],
        "layout": {}
    }
    if domain == "analog":
        # Continuous-time impulse response
        if current_poles or current_zeros:
            # Determine simulation time horizon from slowest pole
            slowest = None
            for p in current_poles:
                if p.real < 0:  # stable pole
                    tau = -1.0 / p.real if p.real != 0 else np.inf
                    if slowest is None or tau > slowest:
                        slowest = tau
            if slowest is None:
                slowest = 1.0
            t_end = min(slowest * 5, 100)
            N = 1000 if t_end <= 20 else min(int(1000 * t_end / 20), 5000)
            T = np.linspace(0, t_end, N)
            # Use SciPy impulse response
            try:
                tout, yout = signal.impulse((current_zeros, current_poles, current_k), T=T)
            except Exception:
                # Fallback: if impulse fails (e.g., no poles), define trivial response
                tout = T
                yout = np.full_like(T, current_k)
        else:
            # No poles/zeros: impulse is just k * delta (we'll show one point at t=0)
            tout = np.array([0, 1e-6])
            yout = np.array([current_k, 0.0])
        fig_imp["data"].append({"x": tout, "y": yout, "mode": "lines", "name": "h(t)"})
        fig_imp["layout"] = {
            "title": "Impulse Response",
            "xaxis": {"title": "Time (s)"},
            "yaxis": {"title": "Amplitude"},
            "margin": {"l": 60, "r": 10, "t": 40, "b": 50}
        }
    else:
        # Discrete-time impulse response
        b, a = signal.zpk2tf(current_zeros, current_poles, current_k)
        # Determine length based on pole with largest magnitude
        if current_poles:
            max_mag = max(abs(p) for p in current_poles)
        else:
            max_mag = 0
        if max_mag is None or max_mag < 1:
            # Estimate decay length to 0.1% if IIR
            if current_poles and max_mag > 0:
                N_decay = int(np.ceil(np.log(0.001) / np.log(max_mag)))
                N_samples = max(50, min(N_decay, 1000))
            else:
                # No IIR pole (FIR or no poles)
                N_samples = max(50, len(b))
        else:
            # Poles on or outside unit circle (unlikely for stable filter) – limit length
            N_samples = 1000
        impulse = np.zeros(N_samples)
        impulse[0] = 1.0
        yout = signal.lfilter(b, a, impulse)
        n = np.arange(N_samples)
        fig_imp["data"].append({"x": n, "y": yout, "mode": "lines", "name": "h[n]"})
        fig_imp["layout"] = {
            "title": "Impulse Response",
            "xaxis": {"title": "Samples (n)"},
            "yaxis": {"title": "Amplitude"},
            "margin": {"l": 60, "r": 10, "t": 40, "b": 50}
        }

    return store_data, fig_pz, fig_bode, fig_imp

# Initialize the store with default filter data on startup by triggering the callback
# (This can be done by sending an initial dummy event or simply relying on Dash to call the callback once at startup)


if __name__ == "__main__":
    app.run(debug=True)
