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

DEFAULT_FAMILY  = "Butterworth"
DEFAULT_TYPE    = "low"
DEFAULT_ORDER   = 4
DEFAULT_DOMAIN  = "analog"
DEFAULT_CUTOFF1 = 1.0
DEFAULT_CUTOFF2 = 2.0

FILTER_FAMILIES = [
    {"label":"Butterworth",     "value":"Butterworth"},
    {"label":"Chebyshev I",     "value":"Chebyshev I"},
    {"label":"Chebyshev II",    "value":"Chebyshev II"},
    {"label":"Elliptic",        "value":"Elliptic"},
    {"label":"Bessel",          "value":"Bessel"},
    {"label":"Custom (manual)", "value":"Custom"},
]
FILTER_TYPES = [
    {"label":"Lowpass","value":"low"},
    {"label":"Highpass","value":"high"},
    {"label":"Bandpass","value":"bandpass"},
    {"label":"Bandstop","value":"bandstop"}
]
DOMAIN_OPTIONS = [
    {"label":"Analog","value":"analog"},
    {"label":"Digital","value":"digital"}
]

##################################################
# Initialize the Dash App
##################################################

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name":"viewport","content":"width=device-width,initial-scale=1"}],
)
app.title = "Signal Filter Visualization (LIGO)"

# Inline CSS / layout
app.index_string = r"""
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <!-- Lato -->
    <link href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap" rel="stylesheet">
    <style>
    body {
      font-family:'Lato',sans-serif; margin:0; padding:0; background:#fff;
    }
    .footer {
      text-align:center; padding:10px; border-top:1px solid #eaeaea; background:#f8f9fa; color:#666;
    }
    @media(min-width:992px){
      .main-content-row {
        height:calc(100vh - 210px)!important; overflow:hidden;
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
# Layout
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

domain_radio = dbc.RadioItems(
    id="domain-radio", className="btn-group",
    inputClassName="btn-check", labelClassName="btn btn-outline-primary", labelCheckedClassName="active",
    options=DOMAIN_OPTIONS, value=DEFAULT_DOMAIN
)
domain_radio_group= html.Div(domain_radio, className="btn-group me-2", **{"role":"group"})

family_dd= dcc.Dropdown(id="family-dropdown", options=FILTER_FAMILIES, value=DEFAULT_FAMILY, clearable=False, style={"minWidth":"120px"})
type_dd  = dcc.Dropdown(id="type-dropdown",   options=FILTER_TYPES,    value=DEFAULT_TYPE,   clearable=False, style={"minWidth":"100px"})
order_in = dbc.Input(id="order-input", type="number", value=DEFAULT_ORDER, min=1, step=1, style={"width":"5ch"})

cut1_label= dbc.InputGroupText(id="cutoff1-label", children="Cutoff 1")
cut1_in   = dbc.Input(id="cutoff1-input", type="number", value=DEFAULT_CUTOFF1, step=0.1, style={"width":"6ch"})
cut1_grp  = dbc.InputGroup([cut1_label, cut1_in], id="cutoff1-group", className="me-2")

cut2_label= dbc.InputGroupText(id="cutoff2-label", children="Cutoff 2")
cut2_in   = dbc.Input(id="cutoff2-input", type="number", value=DEFAULT_CUTOFF2, step=0.1, style={"width":"6ch"})
cut2_grp  = dbc.InputGroup([cut2_label, cut2_in], id="cutoff2-group", style={"display":"none"})

controls_row= dbc.Row([
    dbc.Col(domain_radio_group, width="auto"),
    dbc.Col(family_dd, width="auto"),
    dbc.Col(type_dd,   width="auto"),
    dbc.Col(order_in,  width="auto"),
    dbc.Col(cut1_grp,  width="auto"),
    dbc.Col(cut2_grp,  width="auto")
], align="center", className="g-2 flex-wrap")

controls_collapse= dbc.Collapse(controls_row, id="controls-collapse", is_open=False)
toggle_btn= dbc.Button("Filter Controls", id="controls-toggle-btn", color="secondary", className="d-md-none mb-2")

pz_buttons= html.Div([
    dbc.Button("Add Pole", id="add-pole-btn", color="primary", outline=True, size="sm", className="me-2"),
    dbc.Button("Add Zero", id="add-zero-btn", color="primary", outline=True, size="sm", className="me-2"),
    dbc.Button("Clear", id="clear-btn", color="secondary", size="sm")
], className="mb-2")

pz_graph= dcc.Graph(
    id="pz-plot",
    style={"width":"100%","height":"100%"},
    config={"editable":True,"edits":{"shapePosition":True}, "displayModeBar":False}
)
bode_graph= dcc.Graph(
    id="bode-plot",
    style={"width":"100%","height":"50%"},
    config={"displayModeBar":False}
)
impulse_graph= dcc.Graph(
    id="impulse-plot",
    style={"width":"100%","height":"50%"},
    config={"displayModeBar":False}
)

app.layout= html.Div([
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

    dcc.Store(id="pz-store", data={"poles":[],"zeros":[],"gain":1.0})
], style={"display":"flex","flexDirection":"column","minHeight":"100vh"})


##################################################
# JSON sanitize
##################################################
def sanitize_json(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):
        return [sanitize_json(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    if isinstance(obj, complex):
        if abs(obj.imag)<1e-14:
            return float(obj.real)
        else:
            raise ValueError(f"Truly complex numeric: {obj}")
    if isinstance(obj,(np.float64,np.float32,np.float16,np.int64,np.int32)):
        return float(obj)
    if hasattr(obj,'real') and hasattr(obj,'imag'):
        if abs(obj.imag)<1e-14:
            return float(obj.real)
        else:
            raise ValueError(f"Complex numeric: {obj}")
    if isinstance(obj,(float,int,str,bool,type(None))):
        return obj
    return obj

##################################################
# Filter design
##################################################
def design_filter(family, ftype, order, domain, c1, c2):
    import numpy as np
    from scipy import signal
    analog=(domain=="analog")
    if ftype in ["bandpass","bandstop"]:
        lo=min(c1,c2); hi=max(c1,c2)
        if analog:
            if lo<=0: lo=1e-6
        else:
            if lo<=0: lo=1e-6
            if hi>=1: hi=0.999999
        Wn=[lo,hi]
    else:
        Wn=c1
        if analog:
            if Wn<=0: Wn=1e-6
        else:
            if Wn>=1: Wn=0.999999
            if Wn<=0: Wn=1e-6
    try:
        if family=="Butterworth":
            z,p,k= signal.butter(order,Wn,btype=ftype,analog=analog,output='zpk')
        elif family=="Chebyshev I":
            z,p,k= signal.cheby1(order,1,Wn,btype=ftype,analog=analog,output='zpk')
        elif family=="Chebyshev II":
            z,p,k= signal.cheby2(order,40,Wn,btype=ftype,analog=analog,output='zpk')
        elif family=="Elliptic":
            z,p,k= signal.ellip(order,1,40,Wn,btype=ftype,analog=analog,output='zpk')
        elif family=="Bessel":
            z,p,k= signal.bessel(order,Wn,btype=ftype,analog=analog,output='zpk')
        else:
            z,p,k= np.array([]), np.array([]),1.0
    except:
        z,p,k= np.array([]), np.array([]),1.0

    zeros= [[float(zr.real), float(zr.imag)] for zr in z]
    poles= [[float(pr.real), float(pr.imag)] for pr in p]
    return zeros, poles, float(k)

##################################################
# Toggle Controls
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
# Show/hide second cutoff
##################################################
@app.callback(
    Output("cutoff2-group","style"),
    Output("cutoff1-label","children"),
    Output("cutoff2-label","children"),
    Input("type-dropdown","value"),
    Input("domain-radio","value")
)
def toggle_cut2(ftype, domain):
    if ftype in ["bandpass","bandstop"]:
        st={}
        c1="Low Cutoff"
        c2="High Cutoff"
    else:
        st={"display":"none"}
        c1="Cutoff Freq"
        c2="Cutoff 2"
    if domain=="analog":
        c1+=" (rad/s)"
        c2+=" (rad/s)"
    else:
        c1+=" (norm.)"
        c2+=" (norm.)"
    return st, c1, c2


##################################################
# Single callback
##################################################
@app.callback(
    Output("pz-store","data"),
    Output("pz-plot","figure"),
    Output("bode-plot","figure"),
    Output("impulse-plot","figure"),
    Input("family-dropdown","value"),
    Input("type-dropdown","value"),
    Input("order-input","value"),
    Input("domain-radio","value"),
    Input("cutoff1-input","value"),
    Input("cutoff2-input","value"),
    Input("add-pole-btn","n_clicks"),
    Input("add-zero-btn","n_clicks"),
    Input("clear-btn","n_clicks"),
    Input("pz-plot","relayoutData"),
    State("pz-store","data")
)
def update_all(
    fam, ftype, order, domain, c1, c2,
    addp, addz, clear_btn, relayoutData,
    store_data
):
    ctx=callback_context
    trig_id= ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    print(f"[DEBUG] triggered by => {trig_id}")

    # read store
    old_poles= [complex(p[0],p[1]) for p in store_data["poles"]]
    old_zeros=[complex(z[0],z[1]) for z in store_data["zeros"]]
    old_gain= store_data["gain"]

    # if user changed filter param => re-design if not custom
    if trig_id in [
        "family-dropdown","type-dropdown","order-input",
        "domain-radio","cutoff1-input","cutoff2-input"
    ]:
        if fam!="Custom":
            zlist, plist, k_new= design_filter(fam, ftype, order, domain, c1, c2)
            old_zeros= [complex(zv[0],zv[1]) for zv in zlist]
            old_poles= [complex(pv[0],pv[1]) for pv in plist]
            old_gain= k_new
        else:
            # if domain changed in custom => reset
            if trig_id=="domain-radio":
                old_zeros=[]; old_poles=[]; old_gain=1.0

    # add pole
    if trig_id=="add-pole-btn":
        if domain=="analog":
            idx=len(old_poles)
            old_poles.append(complex(-0.5*(idx+1),0))
        else:
            idx=len(old_poles)
            val=0.5+0.2*idx
            if val>0.9: val=0.9
            old_poles.append(complex(val,0))

    # add zero
    if trig_id=="add-zero-btn":
        if domain=="analog":
            idx=len(old_zeros)
            old_zeros.append(complex(-0.5*(idx+1),0))
        else:
            idx=len(old_zeros)
            if idx==0:
                old_zeros.append(complex(1,0))
            elif idx==1:
                old_zeros.append(complex(-1,0))
            else:
                old_zeros.append(complex(0.5,0))

    # clear
    if trig_id=="clear-btn":
        if fam!="Custom":
            zlist, plist, k_new= design_filter(fam, ftype, order, domain, c1, c2)
            old_zeros= [complex(zv[0],zv[1]) for zv in zlist]
            old_poles= [complex(pv[0],pv[1]) for pv in plist]
            old_gain= k_new
        else:
            old_zeros=[]; old_poles=[]; old_gain=1.0

    # If triggered by pz-plot or at least if relayoutData => parse the shape movement
    if trig_id=="pz-plot":
        if relayoutData and isinstance(relayoutData,dict):
            # You have keys like "shapes[0].x0": val, "shapes[0].name": val, ...
            # Let's gather them by shape index
            shapes_by_idx= {}
            for k,v in relayoutData.items():
                if k.startswith("shapes["):
                    # parse shapes[IDX].attr
                    shape_idx_str= k.split("[")[1].split("]")[0]  # "0" or "1"
                    shape_idx= int(shape_idx_str)
                    attr= k.split(".")[-1]  # x0, x1, y0, y1, name, ...
                    if shape_idx not in shapes_by_idx:
                        shapes_by_idx[shape_idx]={}
                    shapes_by_idx[shape_idx][attr]= v

            # Now unify them => shape i => { "name":..., "x0":..., "x1":..., "y0":..., "y1":...}
            zero_map= {}
            pole_lines= {}

            for i, shape_dict in shapes_by_idx.items():
                sname= shape_dict.get("name","")
                if not sname or sname=="stable-region":
                    continue
                # gather coords
                x0= shape_dict.get("x0", 0.0)
                x1= shape_dict.get("x1", 0.0)
                y0= shape_dict.get("y0", 0.0)
                y1= shape_dict.get("y1", 0.0)
                cx= (x0 + x1)/2.0
                cy= (y0 + y1)/2.0

                if sname.startswith("zero-"):
                    zidx= int(sname.split("-")[1])
                    zero_map[zidx]= (cx, cy)
                elif sname.startswith("pole-"):
                    # e.g. pole-2-line0
                    parts= sname.split("-")
                    pidx= int(parts[1])
                    if pidx not in pole_lines:
                        pole_lines[pidx]= []
                    pole_lines[pidx].append( (cx, cy) )

            # unify zeros
            for zidx,(xx,yy) in zero_map.items():
                old_zeros[zidx]= complex(xx,yy)

            # unify poles => if line0 or line1 missing => old store
            for pidx in range(len(old_poles)):
                if pidx not in pole_lines:
                    continue
                coords= pole_lines[pidx]
                if len(coords)==1:
                    # user only dragged one line => get old store location for the other
                    old_cx= old_poles[pidx].real
                    old_cy= old_poles[pidx].imag
                    coords.append((old_cx, old_cy))
                # average
                xs= [c[0] for c in coords]
                ys= [c[1] for c in coords]
                cx_avg= sum(xs)/len(xs)
                cy_avg= sum(ys)/len(ys)
                old_poles[pidx]= complex(cx_avg, cy_avg)

    # Finally recalc bode/impulse
    def freqz_zpk(zlist, plist, k):
        analog=(domain=="analog")
        if analog:
            if ftype in ["bandpass","bandstop"]:
                lo=min(c1,c2)
                hi=max(c1,c2)
                fmin=max(1e-3, 0.1*lo)
                fmax=max(fmin*10, 10*hi)
            else:
                fmin=max(1e-3,0.1*c1)
                fmax=max(fmin*10,10*c1)
            w= np.logspace(np.log10(fmin), np.log10(fmax),500)
            s= 1j*w
            num= np.ones_like(s,dtype=complex)
            den= np.ones_like(s,dtype=complex)
            for zz in zlist:
                num*= (s-zz)
            for pp in plist:
                den*= (s-pp)
            H= k* num/den
            return w,H
        else:
            w= np.linspace(0,np.pi,800)
            ejw= np.exp(1j*w)
            num= np.ones_like(ejw, dtype=complex)
            den= np.ones_like(ejw, dtype=complex)
            for zz in zlist:
                num*= (ejw-zz)
            for pp in plist:
                den*= (ejw-pp)
            H= k* num/den
            return w,H

    w,H= freqz_zpk(old_zeros, old_poles, old_gain)
    mag= 20.*np.log10(np.abs(H)+1e-12)
    phase= np.unwrap(np.angle(H))
    phase_deg= phase*180./np.pi

    bode_fig={
        "data":[
            {"x": w.tolist(),"y": mag.tolist(),
             "mode":"lines","name":"Mag(dB)",
             "marker":{"color":LIGO_PURPLE},"yaxis":"y1"
            },
            {"x": w.tolist(),"y": phase_deg.tolist(),
             "mode":"lines","name":"Phase(deg)",
             "marker":{"color":"#ff7f0e"},"yaxis":"y2"
            }
        ],
        "layout":{
            "title":"Frequency Response (Bode Plot)",
            "margin":{"l":60,"r":60,"t":40,"b":50},
            "showlegend":False,
            "xaxis":{
                "title":"Frequency (rad/s)" if domain=="analog" else "Frequency (rad/sample)"
            },
            "yaxis":{"title":"Magnitude (dB)"},
            "yaxis2":{"title":"Phase (deg)","overlaying":"y","side":"right"}
        }
    }
    if domain=="analog":
        bode_fig["layout"]["xaxis"]["type"]="log"

    # impulse
    impulse_fig={
        "data":[],
        "layout":{
            "title":"Impulse Response",
            "margin":{"l":60,"r":20,"t":40,"b":50}
        }
    }

    if domain=="analog":
        if not old_poles and not old_zeros:
            t_=[0,1e-3]
            h_=[old_gain,0]
        else:
            neg_p= [p for p in old_poles if p.real<0]
            if neg_p:
                slowest= max([-1.0/p.real for p in neg_p if p.real<0], default=1)
            else:
                slowest=1
            tmax= min(slowest*5,100)
            t= np.linspace(0,tmax,500)
            b,a= signal.zpk2tf(old_zeros, old_poles, old_gain)
            try:
                tout,yout= signal.impulse((b,a), T=t)
                t_= tout; h_= yout
            except:
                t_= t; h_= np.zeros_like(t)
        impulse_fig["data"].append({"x":t_,"y":h_,"mode":"lines","name":"h(t)"})
        impulse_fig["layout"]["xaxis"]={"title":"Time (s)"}
        impulse_fig["layout"]["yaxis"]={"title":"Amplitude"}
    else:
        b,a= signal.zpk2tf(old_zeros, old_poles, old_gain)
        if old_poles:
            max_mag= max(abs(pp) for pp in old_poles)
        else:
            max_mag= 0
        N=100 if max_mag<1 else 200
        imp= np.zeros(N); imp[0]=1.
        h_= signal.lfilter(b,a,imp)
        n_= np.arange(N)
        impulse_fig["data"].append({"x":n_.tolist(),"y":h_.tolist(),"mode":"lines","name":"h[n]"})
        impulse_fig["layout"]["xaxis"]={"title":"Samples (n)"}
        impulse_fig["layout"]["yaxis"]={"title":"Amplitude"}

    # build PZ figure
    fig_pz={
        "data":[],
        "layout":{
            "title":"Pole-Zero Plot",
            "uirevision":"pz-uirev",  # preserve shape changes
            "xaxis":{"title":"Real Axis"},
            "yaxis":{"title":"Imag Axis","scaleanchor":"x","scaleratio":1},
            "margin":{"l":60,"r":20,"t":40,"b":50},
            "shapes":[],
            "showlegend":False
        }
    }

    # stable region
    if domain=="analog":
        shape_stable={
            "type":"rect","xref":"x","yref":"y","name":"stable-region",
            "x0":-9999,"x1":0,"y0":-9999,"y1":9999,
            "fillcolor":"rgba(0,255,0,0.07)",
            "line":{"width":0},"layer":"below"
        }
    else:
        shape_stable={
            "type":"circle","xref":"x","yref":"y","name":"stable-region",
            "x0":-1,"x1":1,"y0":-1,"y1":1,
            "fillcolor":"rgba(0,255,0,0.07)",
            "line":{"width":0},"layer":"below"
        }
    fig_pz["layout"]["shapes"].append(shape_stable)

    all_x= [z.real for z in old_zeros]+[p.real for p in old_poles]
    all_y= [z.imag for z in old_zeros]+[p.imag for p in old_poles]
    ax_lim=1.0
    if all_x or all_y:
        maxi= max([1.0]+[abs(v) for v in all_x+all_y])
        ax_lim= maxi*1.2
    fig_pz["layout"]["xaxis"]["range"]=[-ax_lim, ax_lim]
    fig_pz["layout"]["yaxis"]["range"]=[-ax_lim, ax_lim]

    # zeros => shape
    for zidx,z_ in enumerate(old_zeros):
        zx=float(z_.real)
        zy=float(z_.imag)
        shape_z={
            "type":"circle","xref":"x","yref":"y","name":f"zero-{zidx}",
            "x0":zx-0.05,"x1":zx+0.05,
            "y0":zy-0.05,"y1":zy+0.05,
            "line":{"color":"#1f77b4","width":2},
            "fillcolor":"rgba(0,0,0,0)"
        }
        fig_pz["layout"]["shapes"].append(shape_z)

    # poles => 2 lines each
    for pidx,p_ in enumerate(old_poles):
        px=float(p_.real)
        py=float(p_.imag)
        d=0.07
        shape_l1={
            "type":"line","xref":"x","yref":"y","name":f"pole-{pidx}-line0",
            "x0":px-d,"x1":px+d,
            "y0":py-d,"y1":py+d,
            "line":{"color":"#d62728","width":2}
        }
        shape_l2={
            "type":"line","xref":"x","yref":"y","name":f"pole-{pidx}-line1",
            "x0":px-d,"x1":px+d,
            "y0":py+d,"y1":py-d,
            "line":{"color":"#d62728","width":2}
        }
        fig_pz["layout"]["shapes"].append(shape_l1)
        fig_pz["layout"]["shapes"].append(shape_l2)

    # finalize store
    new_store={
        "poles": [[p.real,p.imag] for p in old_poles],
        "zeros": [[z.real,z.imag] for z in old_zeros],
        "gain": old_gain
    }

    fig_pz= sanitize_json(fig_pz)
    bode_fig= sanitize_json(bode_fig)
    impulse_fig= sanitize_json(impulse_fig)

    return new_store, fig_pz, bode_fig, impulse_fig


if __name__=="__main__":
    app.run(debug=True)
