import numpy as np
from numpy import pi
from scipy import signal
import dash
from dash import dcc, html, Input, Output, State, callback, exceptions

app = dash.Dash(__name__, suppress_callback_exceptions=True)

GRID_STEP = 0.1  # snapping step for dragging poles/zeros

def design_filter(filter_family, filter_type, order, cutoff1, cutoff2=None,
                  ripple1=0.1, ripple2=20.0, analog=True):
    """Uses scipy.signal design routines to produce z, p, k for the specified filter parameters."""
    if analog:
        Wn = cutoff1 if filter_type in ['lowpass', 'highpass'] else [cutoff1, cutoff2]
    else:
        Wn = cutoff1 if filter_type in ['lowpass', 'highpass'] else [cutoff1, cutoff2]

    if filter_family == 'Butterworth':
        z, p, k = signal.butter(order, Wn, btype=filter_type, analog=analog, output='zpk')
    elif filter_family == 'Chebyshev I':
        z, p, k = signal.cheby1(order, ripple1, Wn, btype=filter_type, analog=analog, output='zpk')
    elif filter_family == 'Chebyshev II':
        z, p, k = signal.cheby2(order, ripple2, Wn, btype=filter_type, analog=analog, output='zpk')
    elif filter_family == 'Elliptic':
        z, p, k = signal.ellip(order, ripple1, ripple2, Wn, btype=filter_type, analog=analog, output='zpk')
    elif filter_family == 'Bessel':
        if analog:
            z, p, k = signal.bessel(order, Wn, btype=filter_type, analog=True, output='zpk')
        else:
            # design analog bessel then bilinear transform
            z_a, p_a, k_a = signal.bessel(order, cutoff1, btype=filter_type, analog=True, output='zpk')
            z, p, k = signal.bilinear_zpk(z_a, p_a, k_a, fs=2.0)
    else:
        z, p, k = signal.butter(order, Wn, btype=filter_type, analog=analog, output='zpk')

    return np.array(z), np.array(p), k


def compute_frequency_response(z, p, k, analog=True):
    """Compute the Bode magnitude (dB) and phase (deg) vs frequency."""
    if analog:
        freqs = []
        if len(p) > 0:
            freqs += list(np.abs(p))
        if len(z) > 0:
            freqs += list(np.abs(z))
        freqs = [f for f in freqs if f != 0]
        if not freqs:
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

    mag = 20 * np.log10(np.clip(np.abs(H), 1e-12, None))
    phase = np.unwrap(np.angle(H))
    phase_deg = phase * 180.0 / np.pi
    return freq_axis, mag, phase_deg


def format_transfer_function(z, p, k, analog=True):
    """
    Format the filter's transfer function in a LaTeX-like string plus a summary of zeros/poles/gain.
    Ensures small imaginary parts are forced to real so we don't get TypeErrors comparing complex to int.
    """
    num_poly = np.poly(z) * k
    den_poly = np.poly(p)
    # Ensure they are arrays
    num_poly = np.atleast_1d(num_poly)
    den_poly = np.atleast_1d(den_poly)

    def poly_to_string(coeffs, var='s'):
        """Convert polynomial coefficients into a string, e.g. '1.00 s^2 + 0.5 s + 0.25'.
           Skip near-zero coefficients. For truly complex coeffs, just show '(x + yj)' without sign logic.
        """
        coeffs = np.atleast_1d(coeffs)
        N = len(coeffs)
        terms = []

        for i, c in enumerate(coeffs):
            # Attempt to force near-real if imaginary part is tiny
            c = np.real_if_close(c, tol=1e-12)
            if isinstance(c, np.ndarray):
                # real_if_close might produce a 0D array
                c = c.item()

            # if it's effectively zero, skip
            if abs(c) < 1e-12:
                continue

            power = N - i - 1

            # If c is still complex with non-negligible imaginary part, handle separately
            if isinstance(c, complex) and abs(c.imag) >= 1e-12:
                # We'll just do something like (x+/-yj) with no sign logic
                c_str = f"({c.real:.4g}{'+' if c.imag >=0 else '−'}{abs(c.imag):.4g}j)"
                term = _format_term(c_str, power, var)
                # For the sign, let's just treat it as + unless it's the first term
                # We'll unify all terms with explicit +/− if you prefer, but let's do simpler:
                if i == 0:
                    term_str = term  # no leading sign for first
                else:
                    term_str = f" + {term}"
            else:
                # Now c should be real or near-real
                c_val = float(c)
                sign = '+' if c_val >= 0 else '−'
                c_abs_str = f"{abs(c_val):.4g}"
                term = _format_term(c_abs_str, power, var)
                if i == 0:
                    # first term
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

    def _format_term(coeff_str, power, var):
        """Helper to attach s^power or z^power."""
        if power == 0:
            return coeff_str
        elif power == 1:
            return f"{coeff_str}{var}"
        else:
            return f"{coeff_str}{var}^{power}"

    var = 's' if analog else 'z'
    num_str = poly_to_string(num_poly, var=var)
    den_str = poly_to_string(den_poly, var=var)
    tf_latex = f"H({var}) = \\frac{{{num_str}}}{{{den_str}}}"

    # Build the text details for zeros/poles
    z_array = np.atleast_1d(z)
    p_array = np.atleast_1d(p)

    def format_complex_val(c):
        # If near-real, show as real, else show real±imag
        c = np.real_if_close(c, tol=1e-12)
        if isinstance(c, np.ndarray):
            c = c.item()
        if isinstance(c, complex) and abs(c.imag) >= 1e-12:
            re = c.real
            im = c.imag
            sign_im = '+' if im >= 0 else '−'
            return f"{re:.3f}{sign_im}{abs(im):.3f}j"
        else:
            # treat as float
            return f"{float(c):.3f}"

    if len(z_array) == 0:
        z_str = "None"
    else:
        z_str = ", ".join(format_complex_val(val) for val in z_array)

    if len(p_array) == 0:
        p_str = "None"
    else:
        p_str = ", ".join(format_complex_val(val) for val in p_array)

    details = f"Zeros: {z_str}; Poles: {p_str}; Gain: {k:.4g}"
    return tf_latex, details


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
        html.Div(id='cutoff-note', style={'fontSize':'12px','fontStyle':'italic'}),

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
    """Single callback that updates the stored poles and zeros whenever
       filter parameters change, or user manipulates them directly."""
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if not zpk_data:
        zpk_data = {'zeros': [], 'poles': [], 'gain': 1.0, 'analog': True}

    analog = (analog_dig == 'analog')
    z_list = zpk_data['zeros']
    p_list = zpk_data['poles']
    k = zpk_data['gain']

    # If a filter parameter changed, re-design from scratch
    filter_param_ids = [
        'filter-family', 'filter-type', 'analog-digital', 'filter-order',
        'cutoff1', 'cutoff2', 'ripple1', 'ripple2'
    ]
    if triggered_id in filter_param_ids:
        if filt_type in ["bandpass", "bandstop"] and c2 < c1:
            c1, c2 = c2, c1

        if not analog:  # digital mode
            # clamp or validate c1, c2
            c1 = max(1e-6, min(c1, 0.999999))
            if c2 is not None:
                c2 = max(1e-6, min(c2, 0.999999))

        z, p, k_new = design_filter(
            filter_family=filt_fam, filter_type=filt_type, order=order,
            cutoff1=c1, cutoff2=c2,
            ripple1=rip1 or 0.1, ripple2=rip2 or 30,
            analog=analog
        )
        z_list = [(float(np.real(val)), float(np.imag(val))) for val in z]
        p_list = [(float(np.real(val)), float(np.imag(val))) for val in p]
        k = k_new

    # Add/remove buttons
    if triggered_id == 'add-pole':
        if analog:
            p_list.append((-0.5, 0.0))
        else:
            p_list.append((0.5, 0.0))
    elif triggered_id == 'add-zero':
        if analog:
            z_list.append((0.0, 0.0))
        else:
            z_list.append((1.0, 0.0))
    elif triggered_id == 'remove-pole':
        if p_list:
            p_list.pop()
    elif triggered_id == 'remove-zero':
        if z_list:
            z_list.pop()

    # Dragging shapes
    if triggered_id == 'pz-plot' and relayoutData:
        new_z = list(z_list)
        new_p = list(p_list)
        num_zeros = len(z_list)

        for key, val in relayoutData.items():
            if key.startswith('shapes['):
                idx = int(key.split('[')[1].split(']')[0])
                attr = key.split('.')[-1]
                # shape[0] is boundary (imag axis or unit circle)
                if idx == 0:
                    continue

                # if it's a zero shape
                if 1 <= idx <= num_zeros:
                    z_idx = idx - 1
                    if attr in ('x0', 'y0'):
                        x0 = relayoutData.get(f'shapes[{idx}].x0', new_z[z_idx][0])
                        x1 = relayoutData.get(f'shapes[{idx}].x1', new_z[z_idx][0])
                        y0 = relayoutData.get(f'shapes[{idx}].y0', new_z[z_idx][1])
                        y1 = relayoutData.get(f'shapes[{idx}].y1', new_z[z_idx][1])
                        cx = (x0 + x1)/2.0
                        cy = (y0 + y1)/2.0
                        cx = round(cx / GRID_STEP)*GRID_STEP
                        cy = round(cy / GRID_STEP)*GRID_STEP
                        new_z[z_idx] = (cx, cy)
                else:
                    # pole shape
                    idx_pole_shape = idx - (num_zeros + 1)
                    pole_idx = idx_pole_shape // 2
                    if attr in ('x0', 'y0'):
                        x0 = relayoutData.get(f'shapes[{idx}].x0', new_p[pole_idx][0])
                        x1 = relayoutData.get(f'shapes[{idx}].x1', new_p[pole_idx][0])
                        y0 = relayoutData.get(f'shapes[{idx}].y0', new_p[pole_idx][1])
                        y1 = relayoutData.get(f'shapes[{idx}].y1', new_p[pole_idx][1])
                        cx = (x0 + x1)/2.0
                        cy = (y0 + y1)/2.0
                        cx = round(cx / GRID_STEP)*GRID_STEP
                        cy = round(cy / GRID_STEP)*GRID_STEP
                        new_p[pole_idx] = (cx, cy)

        z_list = new_z
        p_list = new_p

    return {'zeros': z_list, 'poles': p_list, 'gain': k, 'analog': analog}


@callback(
    Output('pz-plot', 'figure'),
    Output('bode-plot', 'figure'),
    Output('tf-latex', 'children'),
    Output('tf-numeric', 'children'),
    Input('zpk-store', 'data')
)
def update_visuals(zpk_data):
    """Rebuild the pole-zero and Bode plots plus TF text from zpk_store."""
    if not zpk_data:
        raise exceptions.PreventUpdate

    analog = zpk_data['analog']
    z_list = zpk_data['zeros']
    p_list = zpk_data['poles']
    k = zpk_data['gain']

    z = np.array([complex(x, y) for x,y in z_list])
    p = np.array([complex(x, y) for x,y in p_list])

    freq_axis, mag, phase_deg = compute_frequency_response(z, p, k, analog=analog)
    bode_fig = {
        'data': [
            {
                'x': freq_axis,
                'y': mag,
                'name': 'Magnitude (dB)',
                'mode': 'lines',
                'line': {'color': 'blue'}
            },
            {
                'x': freq_axis,
                'y': phase_deg,
                'name': 'Phase (deg)',
                'mode': 'lines',
                'line': {'color': 'red'},
                'yaxis': 'y2'
            }
        ],
        'layout': {
            'title': 'Bode Plot',
            'xaxis': {
                'title': 'Frequency (rad/s)' if analog else 'Frequency (rad)',
                'type': 'log'
            },
            'yaxis': {'title': 'Magnitude (dB)'},
            'yaxis2': {
                'title': 'Phase (deg)',
                'overlaying': 'y',
                'side': 'right'
            },
            'legend': {'x': 0.8, 'y': 1.15}
        }
    }

    # Pole-Zero Plot
    shapes = []
    # boundary
    if analog:
        shapes.append({
            'type': 'line', 'x0':0, 'x1':0, 'y0':-10, 'y1':10,
            'line': {'color':'black', 'width':1, 'dash':'dash'}
        })
    else:
        shapes.append({
            'type': 'circle', 'xref':'x','yref':'y',
            'x0':-1,'y0':-1,'x1':1,'y1':1,
            'line': {'color':'black','width':1,'dash':'dot'},
            'fillcolor':'rgba(0,0,0,0)'
        })

    # zeros -> circles
    for z_i in z:
        shapes.append({
            'type': 'circle','xref':'x','yref':'y',
            'x0': z_i.real-0.05, 'x1': z_i.real+0.05,
            'y0': z_i.imag-0.05, 'y1': z_i.imag+0.05,
            'line': {'color':'blue','width':2}
        })
    # poles -> X lines
    for p_i in p:
        cx, cy = p_i.real, p_i.imag
        d = 0.07
        shapes.append({
            'type': 'line','xref':'x','yref':'y',
            'x0': cx-d, 'x1': cx+d, 'y0': cy-d, 'y1': cy+d,
            'line': {'color':'red','width':2}
        })
        shapes.append({
            'type': 'line','xref':'x','yref':'y',
            'x0': cx-d, 'x1': cx+d, 'y0': cy+d, 'y1': cy-d,
            'line': {'color':'red','width':2}
        })

    all_x = [val.real for val in np.concatenate((z, p))] or [0]
    all_y = [val.imag for val in np.concatenate((z, p))] or [0]
    max_val = max([abs(v) for v in (all_x + all_y)] + [1])
    range_lim = max_val*1.2

    pz_layout = {
        'title': 'Pole-Zero Plot',
        'xaxis': {
            'title': 'Real Axis' if analog else 'Re(z)',
            'range': [-range_lim, range_lim]
        },
        'yaxis': {
            'title': 'Imag Axis' if analog else 'Im(z)',
            'range': [-range_lim, range_lim]
        },
        'shapes': shapes,
        'showlegend': False,
        'yaxis_scaleanchor': 'x',
        'yaxis_scaleratio': 1
    }
    pz_fig = {'data': [], 'layout': pz_layout}

    # Transfer Function
    tf_latex, details = format_transfer_function(z, p, k, analog=analog)
    latex_expr = f"$$ {tf_latex} $$"

    return pz_fig, bode_fig, latex_expr, details


if __name__ == '__main__':
    app.run(debug=True)
