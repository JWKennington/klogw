import numpy
from scipy import signal
import dash
from dash import dcc, html, Input, Output, State, callback

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define a small grid step for snapping pole/zero movements (e.g., 0.1 in both real and imag directions)
GRID_STEP = 0.1


# Helper function to design filters and return zeros, poles, gain
def design_filter(
    filter_family,
    filter_type,
    order,
    cutoff1,
    cutoff2=None,
    ripple1=0.1,
    ripple2=20,
    analog=True,
):
    """
    Designs a filter using SciPy and returns its zeros, poles, gain (ZPK form).
    - filter_family: one of 'Butterworth', 'Chebyshev I', 'Chebyshev II', 'Elliptic', 'Bessel'.
    - filter_type: 'lowpass', 'highpass', 'bandpass', or 'bandstop'.
    - order: filter order (integer).
    - cutoff1: cutoff frequency (for low/highpass), or low cutoff for bandpass/stop.
    - cutoff2: high cutoff (for bandpass/bandstop; ignored for low/highpass).
    - ripple1: passband ripple (dB) for Chebyshev I and Elliptic.
    - ripple2: stopband ripple/attenuation (dB) for Chebyshev II and Elliptic.
    - analog: True for analog filter design, False for digital.

    Returns (z, p, k) – zeros, poles, gain of the designed filter.
    """
    # Normalize frequency specifications for SciPy. SciPy expects:
    # Analog: frequencies in rad/s (absolute).
    # Digital: 0-1 range where 1 corresponds to Nyquist frequency (π rad, half the sampling rate).
    if analog:
        Wn = cutoff1 if filter_type in ["lowpass", "highpass"] else [cutoff1, cutoff2]
    else:
        # For digital, assume cutoff given as a fraction of Nyquist (0<cutoff<1).
        Wn = cutoff1 if filter_type in ["lowpass", "highpass"] else [cutoff1, cutoff2]
    btype = filter_type  # SciPy uses same strings for band types

    # Choose the appropriate SciPy design function and parameters
    if filter_family == "Butterworth":
        z, p, k = signal.butter(order, Wn, btype=btype, analog=analog, output="zpk")
    elif filter_family == "Chebyshev I":
        # ripple1 is passband ripple in dB
        z, p, k = signal.cheby1(
            order, ripple1, Wn, btype=btype, analog=analog, output="zpk"
        )
    elif filter_family == "Chebyshev II":
        # ripple2 is stopband attenuation in dB
        z, p, k = signal.cheby2(
            order, ripple2, Wn, btype=btype, analog=analog, output="zpk"
        )
    elif filter_family == "Elliptic":
        # ripple1: passband ripple, ripple2: stopband attenuation
        z, p, k = signal.ellip(
            order, ripple1, ripple2, Wn, btype=btype, analog=analog, output="zpk"
        )
    elif filter_family == "Bessel":
        # SciPy's bessel design works for analog only; if digital, design analog and transform.
        if analog:
            z, p, k = signal.bessel(order, Wn, btype=btype, analog=True, output="zpk")
        else:
            # Design an analog Bessel prototype at analog frequency (cutoff1), then bilinear transform
            z, p, k = signal.bessel(
                order, cutoff1, btype=btype, analog=True, output="zpk"
            )
            # Use bilinear transform to convert to discrete (prewarp cutoff1 to digital domain):
            z, p, k = signal.bilinear_zpk(
                z, p, k, fs=2.0
            )  # fs=2 -> Nyquist = 1 (normalize)
    else:
        # Default to Butterworth if unknown (should not happen with valid input)
        z, p, k = signal.butter(order, Wn, btype=btype, analog=analog, output="zpk")
    return numpy.array(z), numpy.array(p), k


# Helper to compute frequency response (magnitude in dB, phase in degrees) for given zpk
def compute_frequency_response(z, p, k, analog=True):
    # Choose frequency range:
    if analog:
        # For analog, we'll plot from a decade below the smallest pole/zero freq to a decade above the largest.
        # Estimate characteristic freq from poles/zeros (if none, use 1)
        freqs = []
        if len(p) > 0:
            freqs += list(numpy.abs(p))  # pole distances from origin (natural frequencies)
        if len(z) > 0:
            # Include finite zeros; zeros at infinity are not represented in z list
            finite_zeros = [zz for zz in z if zz != 0]
            freqs += list(numpy.abs(finite_zeros))
        freqs = [
            f for f in freqs if f != 0
        ]  # ignore zeros at origin (0 frequency) for range calc
        if len(freqs) == 0:
            f_center = 1.0
        else:
            f_center = numpy.mean(freqs)
        f_min = max(0.001, 0.1 * min(freqs) if len(freqs) > 0 else 0.1)
        f_max = 10 * max(freqs) if len(freqs) > 0 else 100
        # Use a log-spaced frequency array
        w = numpy.logspace(numpy.log10(f_min), numpy.log10(f_max), 500)
        # Evaluate H(jw): product of (jw - z) / (jw - p) times k
        jw = 1j * w
        num = numpy.ones_like(jw, dtype=complex)
        den = numpy.ones_like(jw, dtype=complex)
        for zz in z:
            num *= jw - zz
        for pp in p:
            den *= jw - pp
        H = k * num / den
        freq_axis = w  # use rad/s for x-axis
    else:
        # Digital frequency response from 0 to Nyquist (pi rad)
        # Use log spacing in frequency between a small value and pi.
        w = numpy.logspace(numpy.log10(0.001), numpy.log10(numpy.pi), 500)  # 0.001 rad to π rad
        # Evaluate H(e^{jw}) for each w:
        ejw = numpy.exp(1j * w)
        num = numpy.ones_like(ejw, dtype=complex)
        den = numpy.ones_like(ejw, dtype=complex)
        for zz in z:
            num *= ejw - zz
        for pp in p:
            den *= ejw - pp
        H = k * num / den
        freq_axis = w  # frequency in radians (0 to π)
    # Compute magnitude in dB and phase in degrees
    mag = 20 * numpy.log10(
        numpy.clip(numpy.abs(H), 1e-12, None)
    )  # clip tiny values to avoid log10(0)
    phase = numpy.unwrap(numpy.angle(H))
    phase_deg = phase * 180.0 / numpy.pi
    return freq_axis, mag, phase_deg


# Helper to format transfer function as LaTeX and numeric text
def format_transfer_function(z, p, k, analog=True):
    # Compute polynomial coefficients from poles and zeros
    # np.poly gives coefficients of polynomial with given roots (leading coefficient 1).
    num_poly = numpy.poly(
        z
    )  # zeros array may include zeros at 0 -> poly will handle (gives leading coeff 1)
    den_poly = numpy.poly(p)
    # Scale numerator by gain k (so that H(s) = num_poly/den_poly * k becomes k*num_poly/den_poly)
    num_poly = num_poly * k

    # Format polynomial as a string, e.g. 1.00 s^2 + 0.5 s + 0.25
    def poly_to_string(coeffs, var="s"):
        terms = []
        N = len(coeffs)
        for i, coeff in enumerate(coeffs):
            power = N - i - 1
            # Round the coefficient for display
            coeff_str = f"{coeff:.4g}"
            # Skip terms with zero coefficient (after rounding)
            if float(coeff_str) == 0.0:
                continue
            # Determine sign
            sign = "+" if coeff > 0 else "−"
            # Format term
            if power == 0:
                term = f"{coeff_str}"
            elif power == 1:
                term = f"{coeff_str}{var}"
            else:
                term = f"{coeff_str}{var}^{power}"
            # Remove leading '+' sign for the first term
            if i == 0:
                term_str = term if coeff >= 0 else "−" + term.lstrip("−")
            else:
                term_str = f" {sign} {term.lstrip('+-−')}"
            terms.append(term_str)
        if not terms:
            return "0"
        return "".join(terms)

    var = "s" if analog else "z"
    num_str = poly_to_string(num_poly, var=var)
    den_str = poly_to_string(den_poly, var=var)
    # Construct LaTeX string for transfer function
    tf_latex = f"H({var}) = \\frac{{{num_str}}}{{{den_str}}}"
    # Also produce a simple text listing of zeros, poles, gain
    z_list = ", ".join(
        [
            (
                f"{z_i.real:.2f}{'+' if z_i.imag>=0 else '−'}{abs(z_i.imag):.2f}j"
                if z_i.imag != 0
                else f"{z_i.real:.2f}"
            )
            for z_i in z
        ]
    )
    p_list = ", ".join(
        [
            (
                f"{p_i.real:.2f}{'+' if p_i.imag>=0 else '−'}{abs(p_i.imag):.2f}j"
                if p_i.imag != 0
                else f"{p_i.real:.2f}"
            )
            for p_i in p
        ]
    )
    gain_str = f"{k:.3g}"
    details = f"Zeros: {z_list or 'None'}; Poles: {p_list or 'None'}; Gain: {gain_str}"
    return tf_latex, details


# Layout: Dropdowns, sliders, buttons, and graphs
app.layout = html.Div(
    [
        html.H1("Interactive Filter Visualization App"),
        html.Div(
            [
                html.Label("Filter Family:"),
                dcc.Dropdown(
                    id="filter-family",
                    options=[
                        {"label": "Butterworth", "value": "Butterworth"},
                        {"label": "Chebyshev I", "value": "Chebyshev I"},
                        {"label": "Chebyshev II", "value": "Chebyshev II"},
                        {"label": "Elliptic", "value": "Elliptic"},
                        {"label": "Bessel", "value": "Bessel"},
                    ],
                    value="Butterworth",
                    style={"width": "180px"},
                ),
                html.Label("Filter Type:"),
                dcc.Dropdown(
                    id="filter-type",
                    options=[
                        {"label": "Low-pass", "value": "lowpass"},
                        {"label": "High-pass", "value": "highpass"},
                        {"label": "Band-pass", "value": "bandpass"},
                        {"label": "Band-stop", "value": "bandstop"},
                    ],
                    value="lowpass",
                    style={"width": "150px"},
                ),
                html.Label("Analog/Digital:"),
                dcc.RadioItems(
                    id="analog-digital",
                    options=[
                        {"label": "Analog (s-domain)", "value": "analog"},
                        {"label": "Digital (z-domain)", "value": "digital"},
                    ],
                    value="analog",
                    labelStyle={"margin-right": "20px"},
                ),
                html.Label("Filter Order:"),
                dcc.Slider(
                    id="filter-order",
                    min=1,
                    max=10,
                    value=4,
                    marks={i: str(i) for i in range(1, 11)},
                    step=1,
                ),
                html.Label("Cutoff / Band Edges:"),
                html.Div(
                    [
                        dcc.Input(
                            id="cutoff1",
                            type="number",
                            value=1.0,
                            step=0.1,
                            style={"width": "100px"},
                        ),
                        dcc.Input(
                            id="cutoff2",
                            type="number",
                            value=2.0,
                            step=0.1,
                            style={"width": "100px", "margin-left": "10px"},
                        ),
                    ],
                    id="band-edge-inputs",
                    style={"display": "none"},
                ),  # hidden unless bandpass/bandstop
                html.Div(
                    id="cutoff-note", style={"fontSize": "12px", "fontStyle": "italic"}
                ),
                html.Label("Passband Ripple (dB):"),
                dcc.Input(
                    id="ripple1",
                    type="number",
                    value=1.0,
                    step=0.1,
                    style={"width": "80px"},
                ),
                html.Label("Stopband Ripple (dB):", style={"margin-left": "20px"}),
                dcc.Input(
                    id="ripple2",
                    type="number",
                    value=20.0,
                    step=0.5,
                    style={"width": "80px"},
                ),
            ],
            style={"columnCount": 2, "maxWidth": "700px"},
        ),
        html.Hr(),
        # Buttons to add/remove poles/zeros
        html.Div(
            [
                html.Button("Add Pole", id="add-pole", n_clicks=0),
                html.Button(
                    "Add Zero", id="add-zero", n_clicks=0, style={"margin-left": "10px"}
                ),
                html.Button(
                    "Remove Pole",
                    id="remove-pole",
                    n_clicks=0,
                    style={"margin-left": "30px"},
                ),
                html.Button(
                    "Remove Zero",
                    id="remove-zero",
                    n_clicks=0,
                    style={"margin-left": "10px"},
                ),
            ],
            style={"margin-bottom": "10px"},
        ),
        # Graphs: Pole-Zero plot and Bode plots
        html.Div(
            [
                dcc.Graph(
                    id="pz-plot",
                    config={"editable": True, "edits": {"shapePosition": True}},
                    style={
                        "width": "45%",
                        "display": "inline-block",
                        "height": "400px",
                    },
                ),
                dcc.Graph(
                    id="bode-plot",
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "height": "400px",
                    },
                ),
            ]
        ),
        # Transfer function display
        html.Div(
            [
                html.P(
                    id="tf-latex",
                    style={"fontFamily": "Courier, monospace", "fontSize": "18px"},
                ),
                html.P(
                    id="tf-numeric",
                    style={"fontFamily": "Courier, monospace", "fontSize": "14px"},
                ),
            ],
            style={"marginTop": "20px"},
        ),
        # Hidden storage for current poles, zeros, etc.
        dcc.Store(id="zpk-store"),
    ]
)


# Show or hide the second cutoff input based on filter type
@callback(
    Output("band-edge-inputs", "style"),
    Output("cutoff-note", "children"),
    Input("filter-type", "value"),
)
def toggle_band_inputs(filter_type):
    if filter_type in ["bandpass", "bandstop"]:
        # Show two cutoff inputs for bandpass/stop
        return {
            "display": "block",
            "margin-bottom": "10px",
        }, "Enter low and high cutoff frequencies."
    else:
        # Hide the second cutoff input for lowpass/highpass
        return {"display": "none"}, "Enter cutoff frequency."


# Main callback: When filter parameters change, redesign the filter and update everything
@callback(
    Output("zpk-store", "data"),
    Input("filter-family", "value"),
    Input("filter-type", "value"),
    Input("analog-digital", "value"),
    Input("filter-order", "value"),
    Input("cutoff1", "value"),
    Input("cutoff2", "value"),
    Input("ripple1", "value"),
    Input("ripple2", "value"),
)
def update_filter_design(
    filter_family,
    filter_type,
    analog_digital,
    order,
    cutoff1,
    cutoff2,
    ripple1,
    ripple2,
):
    analog = analog_digital == "analog"
    # For bandpass/stop, ensure cutoff1 < cutoff2
    low_cut = cutoff1
    high_cut = cutoff2 if cutoff2 is not None else cutoff1
    if low_cut is None or (filter_type in ["bandpass", "bandstop"] and cutoff2 is None):
        return dash.no_update  # if inputs not ready, do nothing
    if filter_type in ["bandpass", "bandstop"] and high_cut < low_cut:
        # swap if out of order
        low_cut, high_cut = high_cut, low_cut
    # Design the filter and return the ZPK in a serializable form
    z, p, k = design_filter(
        filter_family,
        filter_type,
        order,
        low_cut,
        cutoff2=high_cut,
        ripple1=ripple1 or 0.1,
        ripple2=ripple2 or 30,
        analog=analog,
    )
    # Store as lists for JSON
    z_list = [(float(numpy.real(z_i)), float(numpy.imag(z_i))) for z_i in z]
    p_list = [(float(numpy.real(p_i)), float(numpy.imag(p_i))) for p_i in p]
    return {"zeros": z_list, "poles": p_list, "gain": k, "analog": analog}


# Update the plots (pole-zero plot and Bode plot) and transfer function text whenever zpk-store changes
@callback(
    Output("pz-plot", "figure"),
    Output("bode-plot", "figure"),
    Output("tf-latex", "children"),
    Output("tf-numeric", "children"),
    Input("zpk-store", "data"),
)
def update_plots_text(zpk_data):
    if not zpk_data:
        raise dash.exceptions.PreventUpdate
    # Extract stored ZPK data
    analog = zpk_data.get("analog", True)
    z_list = zpk_data.get("zeros", [])
    p_list = zpk_data.get("poles", [])
    k = zpk_data.get("gain", 1.0)
    # Convert lists back to complex arrays
    z = numpy.array([complex(x, y) for x, y in z_list])
    p = numpy.array([complex(x, y) for x, y in p_list])
    # Compute frequency response for Bode plot
    freq, mag, phase = compute_frequency_response(z, p, k, analog=analog)
    # Prepare the Bode magnitude & phase figure
    bode_fig = {
        "data": [
            {
                "x": freq,
                "y": mag,
                "name": "Magnitude",
                "mode": "lines",
                "line": {"color": "blue"},
            },
            {
                "x": freq,
                "y": phase,
                "name": "Phase",
                "yaxis": "y2",
                "mode": "lines",
                "line": {"color": "red"},
            },
        ],
        "layout": {
            "title": "Bode Plot",
            "xaxis": {
                "title": "Frequency (rad/s)" if analog else "Frequency (rad)",
                "type": "log",
            },
            "yaxis": {"title": "Magnitude (dB)", "rangemode": "tozero"},
            "yaxis2": {"title": "Phase (deg)", "overlaying": "y", "side": "right"},
            "legend": {"x": 0.8, "y": 1.1},
        },
    }
    # Prepare the Pole-Zero plot figure
    # Shapes for poles and zeros
    shapes = []
    # Add stability boundary shape: line for analog, circle for digital
    if analog:
        # vertical line at Re=0 (imag axis)
        shapes.append(
            {
                "type": "line",
                "x0": 0,
                "x1": 0,
                "y0": -10,
                "y1": 10,
                "line": {"color": "black", "width": 1, "dash": "dash"},
            }
        )
    else:
        # unit circle
        shapes.append(
            {
                "type": "circle",
                "xref": "x",
                "yref": "y",
                "x0": -1,
                "y0": -1,
                "x1": 1,
                "y1": 1,
                "line": {"color": "black", "width": 1, "dash": "dot"},
                "fillcolor": "rgba(0,0,0,0)",
            }
        )
    # Add each zero as a small circle shape
    for z_i in z:
        shapes.append(
            {
                "type": "circle",
                "xref": "x",
                "yref": "y",
                "x0": numpy.real(z_i) - 0.05,
                "y0": numpy.imag(z_i) - 0.05,
                "x1": numpy.real(z_i) + 0.05,
                "y1": numpy.imag(z_i) + 0.05,
                "line": {"color": "blue", "width": 2},
            }
        )
    # Add each pole as an "X" (two line shapes crossing)
    for p_i in p:
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": numpy.real(p_i) - 0.07,
                "y0": numpy.imag(p_i) - 0.07,
                "x1": numpy.real(p_i) + 0.07,
                "y1": numpy.imag(p_i) + 0.07,
                "line": {"color": "red", "width": 2},
            }
        )
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": numpy.real(p_i) - 0.07,
                "y0": numpy.imag(p_i) + 0.07,
                "x1": numpy.real(p_i) + 0.07,
                "y1": numpy.imag(p_i) - 0.07,
                "line": {"color": "red", "width": 2},
            }
        )
    # Determine plot ranges (some margin around poles/zeros)
    all_x = [numpy.real(val) for val in numpy.concatenate((z, p))] or [0]
    all_y = [numpy.imag(val) for val in numpy.concatenate((z, p))] or [0]
    max_val = max([abs(val) for val in all_x + all_y] + [1])
    range_lim = max_val * 1.2
    pz_layout = {
        "title": "Pole-Zero Plot",
        "xaxis": {
            "title": "Real Axis" + (" (σ)" if analog else ""),
            "range": [-range_lim, range_lim],
        },
        "yaxis": {
            "title": "Imag Axis" + (" (jω)" if analog else ""),
            "range": [-range_lim, range_lim],
        },
        "shapes": shapes,
        "showlegend": False,
        # Lock aspect ratio so circles appear circular
        "yaxis_scaleanchor": "x",
        "yaxis_scaleratio": 1,
    }
    pz_fig = {
        "data": [],
        "layout": pz_layout,
    }  # no trace data, we use only shapes for markers

    # Format transfer function text
    tf_latex, tf_details = format_transfer_function(z, p, k, analog=analog)
    # Use MathJax in Markdown to render LaTeX
    tf_latex_md = f"$$ {tf_latex} $$"
    return pz_fig, bode_fig, tf_latex_md, tf_details


# Callback for adding/removing poles and zeros
@callback(
    Output("zpk-store", "data"),
    Input("add-pole", "n_clicks"),
    Input("add-zero", "n_clicks"),
    Input("remove-pole", "n_clicks"),
    Input("remove-zero", "n_clicks"),
    State("zpk-store", "data"),
    prevent_initial_call=True,
)
def modify_poles_zeros(
    add_pole_clicks, add_zero_clicks, remove_pole_clicks, remove_zero_clicks, zpk_data
):
    # Determine which button triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered or not zpk_data:
        raise dash.exceptions.PreventUpdate
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    # Get current poles and zeros
    z_list = zpk_data.get("zeros", [])
    p_list = zpk_data.get("poles", [])
    k = zpk_data.get("gain", 1.0)
    analog = zpk_data.get("analog", True)
    # Default new pole/zero placement:
    if analog:
        # analog new pole on negative real axis, new zero at origin (for example)
        new_pole = (-0.5, 0.0)
        new_zero = (0.0, 0.0)
    else:
        # digital new pole inside unit circle on real axis, new zero on unit circle at z=1
        new_pole = (0.5, 0.0)
        new_zero = (1.0, 0.0)
    if trigger_id == "add-pole":
        p_list.append(new_pole)
    elif trigger_id == "add-zero":
        z_list.append(new_zero)
    elif trigger_id == "remove-pole" and p_list:
        p_list.pop()  # remove last pole
    elif trigger_id == "remove-zero" and z_list:
        z_list.pop()  # remove last zero
    else:
        raise dash.exceptions.PreventUpdate
    # Return updated ZPK data
    return {"zeros": z_list, "poles": p_list, "gain": k, "analog": analog}


# Callback to handle drag-and-drop moves of poles/zeros on the pole-zero graph
@callback(
    Output("zpk-store", "data", allow_duplicate=True),
    Input("pz-plot", "relayoutData"),
    State("zpk-store", "data"),
    prevent_initial_call=True,
)
def drag_update(relayoutData, zpk_data):
    if not relayoutData or not zpk_data:
        raise dash.exceptions.PreventUpdate
    # The relayoutData will contain keys like 'shapes[index].x0', 'shapes[index].x1', etc. for moved shapes.
    # We need to identify which shape moved and update the corresponding pole/zero.
    new_z_list = zpk_data.get("zeros", []).copy()
    new_p_list = zpk_data.get("poles", []).copy()
    analog = zpk_data.get("analog", True)
    # Skip if the stability boundary shapes (index 0) is moved (we will ignore changes to the unit circle or imag axis)
    for key, val in relayoutData.items():
        if key.startswith("shapes["):
            # Extract index and attribute
            # Example key: "shapes[3].x0"
            idx = int(key.split("[")[1].split("]")[0])
            attr = key.split(".")[-1]
            # If idx 0 is the stability boundary (imag axis or unit circle), ignore any changes to it by resetting
            if idx == 0:
                return zpk_data  # no change
            # Determine if this index corresponds to a zero or a pole shape:
            # We added shapes in order: 0 is boundary; then each zero as one shape; each pole as two shapes.
            # So shapes indexing: 1..len(z) for zeros, and following that, poles each occupy two indices.
            num_zeros = len(new_z_list)
            # If idx corresponds to a zero shape:
            if 1 <= idx <= num_zeros:
                z_idx = idx - 1
                # For a circle shape, x0,x1,y0,y1 all change together, the center will be (x0+x1)/2, (y0+y1)/2.
                # We update when x0 or y0 is seen (to avoid double updating).
                if attr in ("x0", "y0"):
                    # Compute new center
                    x0 = relayoutData.get(
                        f"shapes[{idx}].x0", new_z_list[z_idx][0] if new_z_list else 0
                    )
                    x1 = relayoutData.get(
                        f"shapes[{idx}].x1", new_z_list[z_idx][0] if new_z_list else 0
                    )
                    y0 = relayoutData.get(
                        f"shapes[{idx}].y0", new_z_list[z_idx][1] if new_z_list else 0
                    )
                    y1 = relayoutData.get(
                        f"shapes[{idx}].y1", new_z_list[z_idx][1] if new_z_list else 0
                    )
                    new_x = (x0 + x1) / 2.0
                    new_y = (y0 + y1) / 2.0
                    # Snap to grid
                    new_x = round(new_x / GRID_STEP) * GRID_STEP
                    new_y = round(new_y / GRID_STEP) * GRID_STEP
                    new_z_list[z_idx] = (new_x, new_y)
            else:
                # It's a pole shape. Poles have two shapes (two lines) for each pole.
                # We can identify which pole by offsetting index by number of zero shapes and grouping by 2.
                idx_pole = idx - (
                    num_zeros + 1
                )  # index among pole shapes (starting at 0)
                pole_idx = idx_pole // 2  # each pole has 2 shapes
                # Only process one of the two line shapes to update the pole (to avoid duplicate handling).
                if attr in ("x0", "y0"):
                    # For line shapes, we take the midpoint as the pole location.
                    x0 = relayoutData.get(
                        f"shapes[{idx}].x0",
                        new_p_list[pole_idx][0] if new_p_list else 0,
                    )
                    x1 = relayoutData.get(
                        f"shapes[{idx}].x1",
                        new_p_list[pole_idx][0] if new_p_list else 0,
                    )
                    y0 = relayoutData.get(
                        f"shapes[{idx}].y0",
                        new_p_list[pole_idx][1] if new_p_list else 0,
                    )
                    y1 = relayoutData.get(
                        f"shapes[{idx}].y1",
                        new_p_list[pole_idx][1] if new_p_list else 0,
                    )
                    new_x = (x0 + x1) / 2.0
                    new_y = (y0 + y1) / 2.0
                    new_x = round(new_x / GRID_STEP) * GRID_STEP
                    new_y = round(new_y / GRID_STEP) * GRID_STEP
                    new_p_list[pole_idx] = (new_x, new_y)
    # Return updated ZPK data with new pole/zero positions
    return {
        "zeros": new_z_list,
        "poles": new_p_list,
        "gain": zpk_data.get("gain", 1.0),
        "analog": analog,
    }


# Run the Dash app (for real deployment, you would use app.run_server)
if __name__ == "__main__":
    app.run(debug=True)
