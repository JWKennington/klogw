import numpy as np
from numpy import pi
from scipy import signal
import dash
from dash import dcc, html, Input, Output, State, callback, exceptions

app = dash.Dash(__name__, suppress_callback_exceptions=True)

GRID_STEP = 0.1  # snapping step for dragging poles/zeros

# Utility: design filter using SciPy, returns z, p, k
def design_filter(filter_family, filter_type, order, cutoff1, cutoff2=None,
                  ripple1=0.1, ripple2=20.0, analog=True):
    if analog:
        Wn = cutoff1 if filter_type in ['lowpass', 'highpass'] else [cutoff1, cutoff2]
    else:
        # For digital, assume the same numeric range 0..1 means fraction of Nyquist
        Wn = cutoff1 if filter_type in ['lowpass', 'highpass'] else [cutoff1, cutoff2]

    btype = filter_type
    if filter_family == 'Butterworth':
        z, p, k = signal.butter(order, Wn, btype=btype, analog=analog, output='zpk')
    elif filter_family == 'Chebyshev I':
        # ripple1 (dB) passband ripple
        z, p, k = signal.cheby1(order, ripple1, Wn, btype=btype, analog=analog, output='zpk')
    elif filter_family == 'Chebyshev II':
        # ripple2 (dB) stopband attenuation
        z, p, k = signal.cheby2(order, ripple2, Wn, btype=btype, analog=analog, output='zpk')
    elif filter_family == 'Elliptic':
        # ripple1: passband ripple, ripple2: stopband attenuation
        z, p, k = signal.ellip(order, ripple1, ripple2, Wn, btype=btype, analog=analog, output='zpk')
    elif filter_family == 'Bessel':
        # SciPy's bessel() for analog only
        if analog:
            z, p, k = signal.bessel(order, Wn, btype=btype, analog=True, output='zpk')
        else:
            # design analog Bessel, then bilinear transform
            z_ana, p_ana, k_ana = signal.bessel(order, cutoff1, btype=btype, analog=True, output='zpk')
            z, p, k = signal.bilinear_zpk(z_ana, p_ana, k_ana, fs=2.0)
    else:
        # default to butter if unknown
        z, p, k = signal.butter(order, Wn, btype=btype, analog=analog, output='zpk')

    return np.array(z), np.array(p), k


# Compute Bode freq response (mag, phase) from ZPK
def compute_frequency_response(z, p, k, analog=True):
    if analog:
        # derive typical freq range from the magnitudes of poles/zeros
        freqs = []
        if len(p) > 0:
            freqs += list(np.abs(p))
        if len(z) > 0:
            finite_zeros = [zz for zz in z if zz != 0]
            freqs += list(np.abs(finite_zeros))
        freqs = [f for f in freqs if f != 0]  # exclude zero/infinite freq
        if len(freqs) == 0:
            f_min, f_max = 0.1, 100
        else:
            f_min = max(0.001, 0.1 * min(freqs))
            f_max = 10 * max(freqs)
        w = np.logspace(np.log10(f_min), np.log10(f_max), 500)

        jw = 1j * w
        num = np.ones_like(jw, dtype=complex)
        den = np.ones_like(jw, dtype=complex)
        for zz in z:
            num *= (jw - zz)
        for pp in p:
            den *= (jw - pp)
        H = k * num / den
        freq_axis = w
    else:
        # for digital, freq range is [0..pi]
        w = np.logspace(np.log10(0.001), np.log10(pi), 500)
        ejw = np.exp(1j * w)
        num = np.ones_like(ejw, dtype=complex)
        den = np.ones_like(ejw, dtype=complex)
        for zz in z:
            num *= (ejw - zz)
        for pp in p:
            den *= (ejw - pp)
        H = k * num / den
        freq_axis = w

    mag = 20*np.log10(np.clip(np.abs(H), 1e-12, None))
    phase = np.unwrap(np.angle(H))
    phase_deg = phase * 180 / np.pi

    return freq_axis, mag, phase_deg


def format_transfer_function(z, p, k, analog=True):
    # Compute polynomial coefficients
    num_poly = np.poly(z) * k
    den_poly = np.poly(p)

    # Force them to be arrays (just in case):
    num_poly = np.atleast_1d(num_poly)
    den_poly = np.atleast_1d(den_poly)

    def poly_to_string(coeffs, var='s'):
        coeffs = np.atleast_1d(coeffs)
        N = len(coeffs)
        terms = []
        for i, coeff in enumerate(coeffs):
            # convert near-real complex to real if possible
            coeff = np.real_if_close(coeff, tol=1e-12)
            if isinstance(coeff, np.ndarray):
                # real_if_close returns 0D array if it is real
                coeff = coeff.item()

            # If there's still a significant imaginary part, handle or skip.
            # For real filters, typically you won't see that if your roots come in conjugate pairs.
            if isinstance(coeff, complex):
                # check magnitude
                if abs(coeff) < 1e-12:
                    continue  # effectively zero
                # If truly complex, you can do something like:
                re = coeff.real
                im = coeff.imag
                # Or just string-format:
                # But let's say you handle them carefully ...
                pass
            else:
                # It's a real float
                if abs(coeff) < 1e-12:
                    continue
            # Format the absolute value for printing (or we keep sign below)
            c_val = coeff
            power = N - i - 1

            # Format sign
            sign = '+' if c_val >= 0 else '−'
            # We'll do a short format
            c_str = f"{abs(c_val):.4g}"

            if power == 0:
                term = c_str
            elif power == 1:
                term = f"{c_str}{var}"
            else:
                term = f"{c_str}{var}^{power}"

            if i == 0:
                # first term with sign
                if c_val < 0:
                    term_str = f"−{term}"
                else:
                    term_str = term
            else:
                term_str = f" {sign} {term}"

            terms.append(term_str)

        if not terms:
            return "0"
        return "".join(terms)

    var = 's' if analog else 'z'
    num_str = poly_to_string(num_poly, var=var)
    den_str = poly_to_string(den_poly, var=var)

    tf_latex = f"H({var}) = \\frac{{{num_str}}}{{{den_str}}}"

    # Summaries
    z_array = np.atleast_1d(z)
    p_array = np.atleast_1d(p)
    z_str = ", ".join([
        f"{z_i.real:.3f}{'+' if z_i.imag>=0 else '−'}{abs(z_i.imag):.3f}j"
        if abs(z_i.imag)>1e-12 else f"{z_i.real:.3f}"
        for z_i in z_array
    ]) or "None"

    p_str = ", ".join([
        f"{p_i.real:.3f}{'+' if p_i.imag>=0 else '−'}{abs(p_i.imag):.3f}j"
        if abs(p_i.imag)>1e-12 else f"{p_i.real:.3f}"
        for p_i in p_array
    ]) or "None"

    details = f"Zeros: {z_str}; Poles: {p_str}; Gain: {k:.4g}"
    return tf_latex, details

# Layout
app.layout = html.Div([
    html.H1("Unified-Callback Signal Processing App"),
    html.Div([
        html.Label("Filter Family:"),
        dcc.Dropdown(
            id='filter-family',
            options=[
                {'label': 'Butterworth', 'value': 'Butterworth'},
                {'label': 'Chebyshev I', 'value': 'Chebyshev I'},
                {'label': 'Chebyshev II', 'value': 'Chebyshev II'},
                {'label': 'Elliptic', 'value': 'Elliptic'},
                {'label': 'Bessel', 'value': 'Bessel'},
            ],
            value='Butterworth',
            style={'width': '180px'}
        ),
        html.Label("Filter Type:"),
        dcc.Dropdown(
            id='filter-type',
            options=[
                {'label': 'Low-pass', 'value': 'lowpass'},
                {'label': 'High-pass', 'value': 'highpass'},
                {'label': 'Band-pass', 'value': 'bandpass'},
                {'label': 'Band-stop', 'value': 'bandstop'}
            ],
            value='lowpass',
            style={'width': '150px'}
        ),
        html.Label("Analog / Digital:"),
        dcc.RadioItems(
            id='analog-digital',
            options=[
                {'label': 'Analog (s-domain)', 'value': 'analog'},
                {'label': 'Digital (z-domain)', 'value': 'digital'}
            ],
            value='analog',
            labelStyle={'margin-right': '20px'}
        ),
        html.Label("Filter Order:"),
        dcc.Slider(
            id='filter-order', min=1, max=10, step=1, value=4,
            marks={i:str(i) for i in range(1,11)}
        ),
        html.Label("Cutoff / Band Edges:"),
        html.Div([
            dcc.Input(id='cutoff1', type='number', value=1.0, step=0.1, style={'width': '100px'}),
            dcc.Input(id='cutoff2', type='number', value=2.0, step=0.1, style={'width': '100px', 'margin-left':'10px'}),
        ], id='band-edge-inputs', style={'display': 'none'}),
        html.Div(id='cutoff-note', style={'fontSize': '12px', 'fontStyle': 'italic'}),

        html.Label("Passband Ripple (dB):"),
        dcc.Input(
            id='ripple1', type='number', value=1.0, step=0.1,
            style={'width': '80px'}
        ),
        html.Label("Stopband Ripple (dB):", style={'margin-left':'20px'}),
        dcc.Input(
            id='ripple2', type='number', value=20.0, step=0.5,
            style={'width': '80px'}
        )
    ], style={'columnCount': 2, 'maxWidth': '700px'}),

    html.Hr(),
    html.Div([
        html.Button("Add Pole", id='add-pole', n_clicks=0),
        html.Button("Add Zero", id='add-zero', n_clicks=0, style={'margin-left':'10px'}),
        html.Button("Remove Pole", id='remove-pole', n_clicks=0, style={'margin-left':'30px'}),
        html.Button("Remove Zero", id='remove-zero', n_clicks=0, style={'margin-left':'10px'})
    ], style={'margin-bottom':'10px'}),

    html.Div([
        dcc.Graph(
            id='pz-plot',
            config={'editable': True, 'edits': {'shapePosition': True}},
            style={'width': '45%', 'display':'inline-block', 'height':'400px'}
        ),
        dcc.Graph(
            id='bode-plot',
            style={'width': '50%', 'display':'inline-block', 'height':'400px'}
        )
    ]),

    html.Div([
        html.P(id='tf-latex', style={'fontSize':'18px', 'fontFamily':'Courier, monospace'}),
        html.P(id='tf-numeric', style={'fontSize':'14px', 'fontFamily':'Courier, monospace'})
    ], style={'marginTop':'20px'}),

    # Hidden store for ZPK data
    dcc.Store(id='zpk-store')
])

# Toggle showing the second cutoff input for bandpass/bandstop
@callback(
    Output('band-edge-inputs', 'style'),
    Output('cutoff-note', 'children'),
    Input('filter-type', 'value')
)
def show_band_edges(filter_type):
    if filter_type in ['bandpass', 'bandstop']:
        return {'display':'block', 'margin-bottom':'10px'}, "Enter low and high cutoff frequencies."
    else:
        return {'display':'none'}, "Enter cutoff frequency."

# ---- Unified Callback for zpk-store ----
@callback(
    Output('zpk-store', 'data'),
    [
        Input('filter-family', 'value'),
        Input('filter-type', 'value'),
        Input('analog-digital', 'value'),
        Input('filter-order', 'value'),
        Input('cutoff1', 'value'),
        Input('cutoff2', 'value'),
        Input('ripple1', 'value'),
        Input('ripple2', 'value'),
        Input('add-pole', 'n_clicks'),
        Input('add-zero', 'n_clicks'),
        Input('remove-pole', 'n_clicks'),
        Input('remove-zero', 'n_clicks'),
        Input('pz-plot', 'relayoutData')
    ],
    State('zpk-store', 'data')
)
def unified_zpk_store_update(
    filt_fam, filt_type, analog_dig, order,
    c1, c2, rip1, rip2,
    addp, addz, remp, remz,
    relayoutData,
    zpk_data
):
    """Single callback to update filter design and pole/zero modifications."""
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # If no existing store, init
    if not zpk_data:
        zpk_data = {'zeros': [], 'poles': [], 'gain': 1.0, 'analog': True}

    analog = (analog_dig == 'analog')
    z_list = zpk_data.get('zeros', [])
    p_list = zpk_data.get('poles', [])
    k = zpk_data.get('gain', 1.0)

    # Filter param IDs that cause a full redesign
    filter_param_ids = [
        'filter-family', 'filter-type', 'analog-digital', 'filter-order',
        'cutoff1', 'cutoff2', 'ripple1', 'ripple2'
    ]
    if triggered_id in filter_param_ids:
        # Redesign from scratch
        low_cut = c1
        high_cut = c2 if c2 is not None else c1
        if filt_type in ['bandpass', 'bandstop'] and high_cut < low_cut:
            low_cut, high_cut = high_cut, low_cut

        z, p, k_new = design_filter(
            filter_family=filt_fam,
            filter_type=filt_type,
            order=order,
            cutoff1=low_cut, cutoff2=high_cut,
            ripple1=rip1 or 0.1, ripple2=rip2 or 30,
            analog=analog
        )
        z_list = [(float(np.real(x)), float(np.imag(x))) for x in z]
        p_list = [(float(np.real(x)), float(np.imag(x))) for x in p]
        k = k_new

    # Add/remove poles/zeros
    if triggered_id == 'add-pole':
        if analog:
            new_p = (-0.5, 0.0)
        else:
            new_p = (0.5, 0.0)
        p_list.append(new_p)

    elif triggered_id == 'add-zero':
        if analog:
            new_z = (0.0, 0.0)
        else:
            new_z = (1.0, 0.0)
        z_list.append(new_z)

    elif triggered_id == 'remove-pole':
        if len(p_list) > 0:
            p_list.pop()

    elif triggered_id == 'remove-zero':
        if len(z_list) > 0:
            z_list.pop()

    # Dragging shapes on pz-plot
    if triggered_id == 'pz-plot' and relayoutData:
        # Re-map shapes to z-list or p-list
        new_z_list = z_list.copy()
        new_p_list = p_list.copy()
        num_zeros = len(new_z_list)
        # shape[0] is the boundary line/circle
        for key, val in relayoutData.items():
            if key.startswith('shapes['):
                idx = int(key.split('[')[1].split(']')[0])
                attr = key.split('.')[-1]
                if idx == 0:
                    # skip boundary shape (imag axis or unit circle)
                    continue
                if 1 <= idx <= num_zeros:
                    # zero shape
                    z_idx = idx - 1
                    if attr in ('x0', 'y0'):
                        x0 = relayoutData.get(f'shapes[{idx}].x0', new_z_list[z_idx][0])
                        x1 = relayoutData.get(f'shapes[{idx}].x1', new_z_list[z_idx][0])
                        y0 = relayoutData.get(f'shapes[{idx}].y0', new_z_list[z_idx][1])
                        y1 = relayoutData.get(f'shapes[{idx}].y1', new_z_list[z_idx][1])
                        xx = (x0 + x1)/2.0
                        yy = (y0 + y1)/2.0
                        xx = round(xx / GRID_STEP)*GRID_STEP
                        yy = round(yy / GRID_STEP)*GRID_STEP
                        new_z_list[z_idx] = (xx, yy)
                else:
                    # pole shape: each pole is two shapes
                    idx_pole_shape = idx - (num_zeros + 1)
                    pole_index = idx_pole_shape // 2
                    if attr in ('x0', 'y0'):
                        x0 = relayoutData.get(f'shapes[{idx}].x0', new_p_list[pole_index][0])
                        x1 = relayoutData.get(f'shapes[{idx}].x1', new_p_list[pole_index][0])
                        y0 = relayoutData.get(f'shapes[{idx}].y0', new_p_list[pole_index][1])
                        y1 = relayoutData.get(f'shapes[{idx}].y1', new_p_list[pole_index][1])
                        xx = (x0 + x1)/2.0
                        yy = (y0 + y1)/2.0
                        xx = round(xx / GRID_STEP)*GRID_STEP
                        yy = round(yy / GRID_STEP)*GRID_STEP
                        new_p_list[pole_index] = (xx, yy)

        z_list = new_z_list
        p_list = new_p_list

    return {
        'zeros': z_list,
        'poles': p_list,
        'gain': k,
        'analog': analog
    }

# --- Callback to build final plots & TF text from updated zpk-store
@callback(
    Output('pz-plot', 'figure'),
    Output('bode-plot', 'figure'),
    Output('tf-latex', 'children'),
    Output('tf-numeric', 'children'),
    Input('zpk-store', 'data')
)
def update_visuals(zpk_data):
    if not zpk_data:
        raise exceptions.PreventUpdate

    analog = zpk_data.get('analog', True)
    z_list = zpk_data.get('zeros', [])
    p_list = zpk_data.get('poles', [])
    k = zpk_data.get('gain', 1.0)

    z = np.array([complex(x, y) for x,y in z_list])
    p = np.array([complex(x, y) for x,y in p_list])

    # Bode
    freq_axis, mag, phase_deg = compute_frequency_response(z, p, k, analog=analog)
    bode_fig = {
        'data': [
            {
                'x': freq_axis, 'y': mag,
                'name': 'Magnitude (dB)', 'mode': 'lines', 'line': {'color': 'blue'}
            },
            {
                'x': freq_axis, 'y': phase_deg,
                'name': 'Phase (deg)', 'mode': 'lines', 'line': {'color': 'red'},
                'yaxis': 'y2'
            }
        ],
        'layout': {
            'title': 'Bode Plot',
            'xaxis': {'title': 'Frequency (rad/s)' if analog else 'Frequency (rad)', 'type': 'log'},
            'yaxis': {'title': 'Magnitude (dB)'},
            'yaxis2': {
                'title': 'Phase (deg)',
                'overlaying': 'y',
                'side': 'right'
            },
            'legend': {'x':0.8, 'y':1.15}
        }
    }

    # PZ plot
    shapes = []
    # Stability boundary
    if analog:
        # vertical line at Re=0
        shapes.append({
            'type': 'line', 'x0':0, 'x1':0, 'y0':-10, 'y1':10,
            'line': {'color':'black', 'width':1, 'dash': 'dash'}
        })
    else:
        # unit circle
        shapes.append({
            'type': 'circle', 'xref':'x', 'yref':'y',
            'x0':-1, 'y0':-1, 'x1':1, 'y1':1,
            'line': {'color':'black', 'width':1, 'dash':'dot'},
            'fillcolor':'rgba(0,0,0,0)'
        })

    # Zeros -> circle
    for z_i in z:
        shapes.append({
            'type': 'circle', 'xref':'x', 'yref':'y',
            'x0': z_i.real - 0.05, 'x1': z_i.real + 0.05,
            'y0': z_i.imag - 0.05, 'y1': z_i.imag + 0.05,
            'line': {'color':'blue', 'width':2}
        })
    # Poles -> X (two line shapes)
    for p_i in p:
        shapes.append({
            'type': 'line', 'xref':'x', 'yref':'y',
            'x0': p_i.real - 0.07, 'x1': p_i.real + 0.07,
            'y0': p_i.imag - 0.07, 'y1': p_i.imag + 0.07,
            'line': {'color':'red', 'width':2}
        })
        shapes.append({
            'type': 'line', 'xref':'x', 'yref':'y',
            'x0': p_i.real - 0.07, 'x1': p_i.real + 0.07,
            'y0': p_i.imag + 0.07, 'y1': p_i.imag - 0.07,
            'line': {'color':'red', 'width':2}
        })

    # Figure bounds
    all_x = [np.real(x) for x in np.concatenate((z, p))] or [0]
    all_y = [np.imag(x) for x in np.concatenate((z, p))] or [0]
    max_val = max([abs(val) for val in (all_x + all_y)] + [1])
    range_lim = max_val * 1.2

    pz_layout = {
        'title': 'Pole-Zero Plot',
        'xaxis': {'title': 'Real Axis' if analog else 'Re(z)', 'range':[-range_lim, range_lim]},
        'yaxis': {'title': 'Imag Axis' if analog else 'Im(z)', 'range':[-range_lim, range_lim]},
        'shapes': shapes,
        'showlegend': False,
        # fix aspect ratio
        'yaxis_scaleanchor': 'x',
        'yaxis_scaleratio': 1
    }
    pz_fig = {'data': [], 'layout': pz_layout}

    # Transfer function
    latex_tf, details = format_transfer_function(z, p, k, analog=analog)
    latex_expr = f"$$ {latex_tf} $$"

    return pz_fig, bode_fig, latex_expr, details

if __name__ == '__main__':
    app.run(debug=True)
