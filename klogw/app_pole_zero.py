import numpy as np
from numpy import pi
from scipy import signal
import dash
from dash import dcc, html, Input, Output, State, callback, exceptions

# ---------- LIGO-inspired color palette -----------
LIGO_PURPLE = '#6F2C8F'    # A purple shade
DARK_BG = '#111111'       # Very dark gray/black
LIGHT_TEXT = '#FFFFFF'    # White

app = dash.Dash(__name__, suppress_callback_exceptions=True)

GRID_STEP = 0.1  # snapping step for dragging poles/zeros

# -------------- Filter design and computations (same as before) ----------------
def design_filter(filter_family, filter_type, order, cutoff1, cutoff2=None,
                  ripple1=0.1, ripple2=20.0, analog=True):
    # ensure band edges are in ascending order if needed
    if filter_type in ['bandpass', 'bandstop'] and cutoff2 is not None and cutoff1 > cutoff2:
        cutoff1, cutoff2 = cutoff2, cutoff1
    if analog:
        Wn = cutoff1 if filter_type in ['lowpass', 'highpass'] else [cutoff1, cutoff2]
    else:
        # clamp digital cutoff(s) to (0,1)
        def clamp(x):
            return max(1e-6, min(0.999999, x))
        if filter_type in ['bandpass', 'bandstop']:
            Wn = [clamp(cutoff1), clamp(cutoff2)]
        else:
            Wn = clamp(cutoff1)

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
            z_a, p_a, k_a = signal.bessel(order, cutoff1, btype=filter_type, analog=True, output='zpk')
            z, p, k = signal.bilinear_zpk(z_a, p_a, k_a, fs=2.0)
    else:
        z, p, k = signal.butter(order, Wn, btype=filter_type, analog=analog, output='zpk')

    return np.array(z), np.array(p), k

def compute_frequency_response(z, p, k, analog=True):
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

def compute_impulse_response(z, p, k, analog=True, n_points=50):
    b, a = signal.zpk2tf(z, p, k)
    if analog:
        sys_analog = signal.lti(b, a)
        tmax = 5.0
        if len(p) > 0:
            reals = [pp.real for pp in p if pp.real < 0]
            if reals:
                tau = -1.0 / min(reals)
                tmax = 5.0 * tau if tau > 0 else 5.0
                if tmax < 0.5:
                    tmax = 0.5
        t = np.linspace(0, tmax, 500)
        tout, h = signal.impulse(sys_analog, T=t)
        return tout, h
    else:
        sys_digital = signal.dlti(b, a)
        tout, h = signal.dimpulse(sys_digital, n=n_points)
        h0 = np.array(h[0]).flatten()
        t0 = np.array(tout)
        return t0, h0

def format_transfer_function(z, p, k, analog=True):
    num_poly = np.poly(z) * k
    den_poly = np.poly(p)
    num_poly = np.atleast_1d(num_poly)
    den_poly = np.atleast_1d(den_poly)

    def poly_to_string(coeffs, var='s'):
        coeffs = np.atleast_1d(coeffs)
        N = len(coeffs)
        terms = []

        def _format_term(c_str, power, var):
            if power == 0:
                return c_str
            elif power == 1:
                return f"{c_str}{var}"
            else:
                return f"{c_str}{var}^{power}"

        for i, c in enumerate(coeffs):
            c = np.real_if_close(c, tol=1e-12)
            if isinstance(c, np.ndarray):
                c = c.item()
            if abs(c) < 1e-12:
                continue
            power = N - i - 1
            if isinstance(c, complex) and abs(c.imag) >= 1e-12:
                re = c.real
                im = c.imag
                sign_im = '+' if im >= 0 else '−'
                c_str = f"({re:.4g}{sign_im}{abs(im):.4g}j)"
                if i == 0:
                    term_str = _format_term(c_str, power, var)
                else:
                    term_str = " + " + _format_term(c_str, power, var)
            else:
                c_val = float(c)
                sign = '+' if c_val >= 0 else '−'
                c_abs_str = f"{abs(c_val):.4g}"
                if i == 0:
                    if c_val < 0:
                        term_str = f"−{_format_term(c_abs_str, power, var)}"
                    else:
                        term_str = _format_term(c_abs_str, power, var)
                else:
                    term_str = f" {sign} {_format_term(c_abs_str, power, var)}"
            terms.append(term_str)

        if not terms:
            return "0"
        return "".join(terms)

    var = 's' if analog else 'z'
    num_str = poly_to_string(num_poly, var=var)
    den_str = poly_to_string(den_poly, var=var)
    tf_latex = f"H({var}) = \\frac{{{num_str}}}{{{den_str}}}"

    def format_complex_val(c):
        c = np.real_if_close(c, tol=1e-12)
        if isinstance(c, np.ndarray):
            c = c.item()
        if isinstance(c, complex) and abs(c.imag) >= 1e-12:
            re = c.real
            im = c.imag
            sign_im = '+' if im >= 0 else '−'
            return f"{re:.3f}{sign_im}{abs(im):.3f}j"
        else:
            return f"{float(c):.3f}"

    z_array = np.atleast_1d(z)
    p_array = np.atleast_1d(p)
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

# ----------------------- LAYOUT with custom LIGO theme -------------------------
app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        <title>LIGO Filter App</title>
        <meta charset="UTF-8">
        <!-- We use a custom body background color below -->
    </head>
    <body style="background-color:{DARK_BG}; color:{LIGHT_TEXT}; margin:0;">
        {{%app_entry%}}
        <footer style="text-align:center; background:{DARK_BG}; color:{LIGHT_TEXT}; margin-top:20px; padding:10px;">
            © Jim Kennington 2025
        </footer>
        {{%config%}}
        {{%scripts%}}
        {{%renderer%}}
    </body>
</html>
"""

app.layout = html.Div(style={'backgroundColor': DARK_BG, 'color': LIGHT_TEXT, 'padding': '10px'}, children=[

    # LIGO logo, top right
    html.Div([
        html.Img(
            src='https://dcc.ligo.org/public/0122/P070084/001/LIGO_logo.jpg',
            style={'height': '50px', 'float': 'right', 'marginRight': '20px'}
        ),
        html.H1("LIGO-Themed Filter Visualization App"),
    ], style={'overflow': 'hidden'}),  # container for logo+title

    html.Div([
        html.Label("Filter Family:", style={'fontWeight': 'bold'}),
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
            style={
                'width': '180px',
                'maxHeight': '150px',     # limit dropdown height
                'overflowY': 'auto',      # scroll if needed
                'color': 'black'          # inside dropdown text color
            }
        ),
        html.Label("Filter Type:", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='filter-type',
            options=[
                {'label': 'Low-pass', 'value': 'lowpass'},
                {'label': 'High-pass', 'value': 'highpass'},
                {'label': 'Band-pass', 'value': 'bandpass'},
                {'label': 'Band-stop', 'value': 'bandstop'}
            ],
            value='lowpass',
            style={
                'width': '150px',
                'maxHeight': '150px',
                'overflowY': 'auto',
                'color': 'black'
            }
        ),
        html.Label("Analog / Digital:", style={'fontWeight': 'bold'}),
        dcc.RadioItems(
            id='analog-digital',
            options=[
                {'label': 'Analog (s-domain)', 'value': 'analog'},
                {'label': 'Digital (z-domain)', 'value': 'digital'}
            ],
            value='analog',
            labelStyle={'margin-right': '20px'},
            style={'display': 'inline-block'}
        ),
        html.Label("Filter Order:", style={'fontWeight': 'bold'}),
        dcc.Slider(
            id='filter-order', min=1, max=10, step=1, value=4,
            marks={i: str(i) for i in range(1, 11)},
            # Dash sliders typically have a white handle by default. We'll keep that.
        ),

        html.Label("Cutoff / Band Edges:", style={'fontWeight': 'bold'}),
        html.Div([
            dcc.Input(
                id='cutoff1', type='number', value=1.0, step=0.1,
                style={'width': '80px', 'marginRight': '5px', 'color': 'black'}
            ),
            dcc.Input(
                id='cutoff2', type='number', value=2.0, step=0.1,
                style={'width': '80px', 'color': 'black'}
            ),
        ], id='band-edge-inputs', style={'display': 'none'}),
        html.Div(id='cutoff-note', style={'fontSize': '12px', 'fontStyle': 'italic'}),

        html.Label("Passband Ripple (dB):", style={'fontWeight': 'bold'}),
        dcc.Input(
            id='ripple1', type='number', value=1.0, step=0.1,
            style={'width': '80px', 'color': 'black'}
        ),
        html.Label("Stopband Ripple (dB):", style={'marginLeft': '20px', 'fontWeight': 'bold'}),
        dcc.Input(
            id='ripple2', type='number', value=20.0, step=0.5,
            style={'width': '80px', 'color': 'black'}
        )
    ], style={'columnCount': 2, 'maxWidth': '700px', 'margin': '10px 0'}),

    html.Hr(style={'border': f'1px solid {LIGO_PURPLE}'}),

    html.Div([
        html.Button("Add Pole", id='add-pole', n_clicks=0, style={'marginRight': '10px', 'backgroundColor': LIGO_PURPLE, 'color': 'white'}),
        html.Button("Add Zero", id='add-zero', n_clicks=0, style={'marginRight': '30px', 'backgroundColor': LIGO_PURPLE, 'color': 'white'}),
        html.Button("Remove Pole", id='remove-pole', n_clicks=0, style={'marginRight': '10px', 'backgroundColor': LIGO_PURPLE, 'color': 'white'}),
        html.Button("Remove Zero", id='remove-zero', n_clicks=0, style={'backgroundColor': LIGO_PURPLE, 'color': 'white'})
    ], style={'marginBottom': '10px'}),

    # MAIN LAYOUT for plots: left = PZ; right = two stacked: Bode (top), Impulse (bottom)
    html.Div([
        html.Div([
            dcc.Graph(
                id='pz-plot',
                config={'editable': True, 'edits': {'shapePosition': True}},
                style={
                    'width': '50vw',   # entire left half
                    'height': '80vh',  # large
                    'display': 'inline-block'
                }
            )
        ], style={'float': 'left'}),

        html.Div([
            # Bode (top)
            dcc.Graph(
                id='bode-plot',
                style={
                    'width': '50vw',
                    'height': '40vh',
                    'display': 'block'
                }
            ),
            # Impulse (bottom)
            dcc.Graph(
                id='impulse-plot',
                style={
                    'width': '50vw',
                    'height': '40vh',
                    'display': 'block'
                }
            )
        ], style={'float': 'right'})
    ], style={'display': 'block', 'clear': 'both'}),

    html.Div([
        html.P(id='tf-latex', style={'fontSize': '18px', 'fontFamily': 'Courier, monospace'}),
        html.P(id='tf-numeric', style={'fontSize': '14px', 'fontFamily': 'Courier, monospace'})
    ], style={'marginTop': '20px'}),

    # Hidden store
    dcc.Store(id='zpk-store')
])

@callback(
    Output('band-edge-inputs', 'style'),
    Output('cutoff-note', 'children'),
    Input('filter-type', 'value')
)
def show_band_edges(filter_type):
    if filter_type in ['bandpass', 'bandstop']:
        return {'display': 'block', 'margin-bottom': '10px'}, "Enter low and high cutoff frequencies."
    else:
        return {'display': 'none'}, "Enter cutoff frequency."

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
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if not zpk_data:
        zpk_data = {'zeros': [], 'poles': [], 'gain': 1.0, 'analog': True}

    analog = (analog_dig == 'analog')
    z_list = list(zpk_data['zeros'])
    p_list = list(zpk_data['poles'])
    k = zpk_data['gain']

    filter_param_ids = [
        'filter-family', 'filter-type', 'analog-digital', 'filter-order',
        'cutoff1', 'cutoff2', 'ripple1', 'ripple2'
    ]
    if triggered_id in filter_param_ids:
        if c1 is None:
            c1 = 1.0
        if (c2 is None) and (filt_type in ['bandpass','bandstop']):
            c2 = 2.0
        # design new filter
        z, p, k_new = design_filter(
            filter_family=filt_fam, filter_type=filt_type, order=order,
            cutoff1=c1, cutoff2=c2,
            ripple1=rip1 or 0.1, ripple2=rip2 or 30,
            analog=analog
        )
        z_list = [(float(np.real(val)), float(np.imag(val))) for val in z]
        p_list = [(float(np.real(val)), float(np.imag(val))) for val in p]
        k = k_new

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

    if triggered_id == 'pz-plot' and relayoutData:
        new_z = z_list.copy()
        new_p = p_list.copy()
        num_zeros = len(z_list)

        for key, val in relayoutData.items():
            if key.startswith('shapes['):
                idx = int(key.split('[')[1].split(']')[0])
                attr = key.split('.')[-1]
                # shape[0] is boundary
                if idx == 0:
                    continue
                if 1 <= idx <= num_zeros:
                    # zero
                    z_idx = idx - 1
                    if attr in ('x0', 'y0'):
                        x0 = relayoutData.get(f'shapes[{idx}].x0', new_z[z_idx][0])
                        x1 = relayoutData.get(f'shapes[{idx}].x1', new_z[z_idx][0])
                        y0 = relayoutData.get(f'shapes[{idx}].y0', new_z[z_idx][1])
                        y1 = relayoutData.get(f'shapes[{idx}].y1', new_z[z_idx][1])
                        cx = round((x0 + x1)/2.0 / GRID_STEP)*GRID_STEP
                        cy = round((y0 + y1)/2.0 / GRID_STEP)*GRID_STEP
                        new_z[z_idx] = (cx, cy)
                else:
                    # pole line
                    idx_pole_shape = idx - (num_zeros + 1)
                    pole_idx = idx_pole_shape // 2
                    if attr in ('x0', 'y0'):
                        x0 = relayoutData.get(f'shapes[{idx}].x0', new_p[pole_idx][0])
                        x1 = relayoutData.get(f'shapes[{idx}].x1', new_p[pole_idx][0])
                        y0 = relayoutData.get(f'shapes[{idx}].y0', new_p[pole_idx][1])
                        y1 = relayoutData.get(f'shapes[{idx}].y1', new_p[pole_idx][1])
                        cx = round((x0 + x1)/2.0 / GRID_STEP)*GRID_STEP
                        cy = round((y0 + y1)/2.0 / GRID_STEP)*GRID_STEP
                        new_p[pole_idx] = (cx, cy)

        z_list = new_z
        p_list = new_p

    return {'zeros': z_list, 'poles': p_list, 'gain': k, 'analog': analog}

@callback(
    Output('pz-plot', 'figure'),
    Output('bode-plot', 'figure'),
    Output('impulse-plot', 'figure'),
    Output('tf-latex', 'children'),
    Output('tf-numeric', 'children'),
    Input('zpk-store', 'data')
)
def update_visuals(zpk_data):
    if not zpk_data:
        raise exceptions.PreventUpdate

    analog = zpk_data['analog']
    z_list = zpk_data['zeros']
    p_list = zpk_data['poles']
    k = zpk_data['gain']

    z = np.array([complex(x, y) for x,y in z_list])
    p = np.array([complex(x, y) for x,y in p_list])

    # Bode
    freq_axis, mag, phase_deg = compute_frequency_response(z, p, k, analog=analog)
    # -- Bode Plot with dark theme styling --
    bode_fig = {
        'data': [
            {
                'x': freq_axis,
                'y': mag,
                'name': 'Magnitude (dB)',
                'mode': 'lines',
                'line': {'color': LIGO_PURPLE}
            },
            {
                'x': freq_axis,
                'y': phase_deg,
                'name': 'Phase (deg)',
                'mode': 'lines',
                'line': {'color': 'orange'},
                'yaxis': 'y2'
            }
        ],
        'layout': {
            'title': f"Bode Plot ({'Analog' if analog else 'Digital'})",
            'paper_bgcolor': DARK_BG,
            'plot_bgcolor': DARK_BG,
            'font': {'color': LIGHT_TEXT},
            'xaxis': {
                'title': 'ω (rad/s)' if analog else 'Ω (rad)',
                'type': 'log',
                'color': LIGHT_TEXT
            },
            'yaxis': {
                'title': 'Magnitude (dB)',
                'color': LIGHT_TEXT
            },
            'yaxis2': {
                'title': 'Phase (deg)',
                'overlaying': 'y',
                'side': 'right',
                'color': LIGHT_TEXT
            },
            'legend': {'x': 0.65, 'y': 1.1}
        }
    }

    # Pole-Zero
    shapes = []
    if analog:
        shapes.append({
            'type': 'line', 'x0': 0, 'x1': 0, 'y0': -10, 'y1': 10,
            'line': {'color': 'gray', 'width': 1, 'dash': 'dash'}
        })
    else:
        shapes.append({
            'type': 'circle', 'xref': 'x','yref': 'y',
            'x0': -1, 'y0': -1, 'x1': 1, 'y1': 1,
            'line': {'color': 'gray','width':1,'dash':'dot'},
            'fillcolor': 'rgba(0,0,0,0)'
        })
    for z_i in z:
        shapes.append({
            'type': 'circle','xref':'x','yref':'y',
            'x0': z_i.real - 0.05, 'x1': z_i.real + 0.05,
            'y0': z_i.imag - 0.05, 'y1': z_i.imag + 0.05,
            'line': {'color': 'blue','width':2}
        })
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
    range_lim = max_val * 1.2

    pz_fig = {
        'data': [],
        'layout': {
            'title': f"Pole-Zero Plot ({'Analog' if analog else 'Digital'})",
            'paper_bgcolor': DARK_BG,
            'plot_bgcolor': DARK_BG,
            'font': {'color': LIGHT_TEXT},
            'xaxis': {
                'title': 'Re(s)' if analog else 'Re(z)',
                'range': [-range_lim, range_lim],
                'color': LIGHT_TEXT
            },
            'yaxis': {
                'title': 'Im(s)' if analog else 'Im(z)',
                'range': [-range_lim, range_lim],
                'color': LIGHT_TEXT,
                'scaleanchor': 'x',
                'scaleratio': 1
            },
            'shapes': shapes,
            'showlegend': False
        }
    }

    # Impulse
    t_imp, h_imp = compute_impulse_response(z, p, k, analog=analog, n_points=50)
    if analog:
        impulse_data = [{
            'x': t_imp,
            'y': h_imp,
            'mode': 'lines',
            'line': {'color': 'lime'},
            'name': 'Impulse (Analog)'
        }]
        impulse_title = "Impulse Response (Analog)"
        x_label = 'Time (s)'
        y_label = 'Amplitude'
    else:
        impulse_data = [{
            'x': t_imp,
            'y': h_imp,
            'mode': 'markers',
            'marker': {'color': 'lime', 'symbol': 'circle', 'size': 6},
            'name': 'Impulse (Digital)'
        }]
        impulse_title = "Impulse Response (Digital)"
        x_label = 'Sample (n)'
        y_label = 'h[n]'

    impulse_fig = {
        'data': impulse_data,
        'layout': {
            'title': impulse_title,
            'paper_bgcolor': DARK_BG,
            'plot_bgcolor': DARK_BG,
            'font': {'color': LIGHT_TEXT},
            'xaxis': {'title': x_label, 'color': LIGHT_TEXT},
            'yaxis': {'title': y_label, 'color': LIGHT_TEXT}
        }
    }

    # TF
    tf_latex, details = format_transfer_function(z, p, k, analog=analog)
    latex_expr = f"$$ {tf_latex} $$"

    return pz_fig, bode_fig, impulse_fig, latex_expr, details


if __name__ == '__main__':
    app.run(debug=True)
