import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
import io
import time
import json
import csv
import random
import imageio
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AnimeGen Pro AI",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
ADMIN_CREDENTIALS = {"admin": "admin"}

def check_login():
    return st.session_state.get("authenticated", False)

def render_login():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    .stApp { background-color:#09090b; }
    .login-wrap {
        max-width:380px; margin:10vh auto 0; padding:2.5rem 2rem;
        background:#111115; border:1px solid #27272a; border-radius:12px;
    }
    .login-title {
        font-family:'IBM Plex Mono',monospace; font-size:1.4rem; color:#f4f4f5;
        text-align:center; margin-bottom:0.25rem; letter-spacing:-0.03em;
    }
    .login-sub {
        font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#52525b;
        text-align:center; margin-bottom:1.8rem;
    }
    </style>
    <div class="login-wrap">
      <div class="login-title">AnimeGen Pro AI</div>
      <div class="login-sub">Admin Access Required</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="admin")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Sign In", use_container_width=True)

    if submitted:
        if ADMIN_CREDENTIALS.get(username) == password:
            st.session_state.authenticated = True
            st.session_state.auth_user = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
def get_theme():
    return st.session_state.get("theme", "dark")

def load_css():
    theme = get_theme()
    with open("style.css") as f:
        css = f.read()
    if theme == "light":
        css += """
        .stApp{background-color:#f4f4f5!important;background-image:none!important;color:#18181b!important;}
        h1{color:#18181b!important;border-bottom-color:#d4d4d8!important;}
        h2,h3{color:#52525b!important;}
        section[data-testid="stSidebar"]{background-color:#e4e4e7!important;border-right-color:#d4d4d8!important;}
        .stButton>button{background:#e4e4e7!important;border-color:#a1a1aa!important;color:#18181b!important;}
        .stButton>button:hover{background:#6366f1!important;border-color:#6366f1!important;color:#fff!important;}
        .stNumberInput input,.stTextInput input,.stSelectbox select{background:#fff!important;border-color:#a1a1aa!important;color:#18181b!important;}
        [data-testid="stMetricValue"]{color:#18181b!important;}
        code{background:#e4e4e7!important;border-color:#d4d4d8!important;color:#6366f1!important;}
        [data-testid="stFileUploader"]{background:#fff!important;border-color:#a1a1aa!important;}
        """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL ARCHITECTURES
# ─────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz,   ngf*8, 4,1,0,bias=False), nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8,ngf*4, 4,2,1,bias=False), nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4,ngf*2, 4,2,1,bias=False), nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2,ngf,   4,2,1,bias=False), nn.BatchNorm2d(ngf),   nn.ReLU(True),
            nn.ConvTranspose2d(ngf,  nc,    4,2,1,bias=False), nn.Tanh()
        )
    def forward(self, x): return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(nc,    ndf,   4,2,1,bias=False), nn.LeakyReLU(0.2,inplace=True)),
            nn.Sequential(nn.Conv2d(ndf,   ndf*2, 4,2,1,bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2,inplace=True)),
            nn.Sequential(nn.Conv2d(ndf*2, ndf*4, 4,2,1,bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2,inplace=True)),
            nn.Sequential(nn.Conv2d(ndf*4, ndf*8, 4,2,1,bias=False), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2,inplace=True)),
        ])
        self.final = nn.Sequential(nn.Conv2d(ndf*8, 1, 4,1,0,bias=False), nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final(x)

    def get_feature_maps(self, x):
        maps, out = [], x
        for layer in self.layers:
            out = layer(out)
            maps.append(out.detach().cpu())
        return maps


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen  = Generator()
    disc = Discriminator()
    ckpt = torch.load("best_gan_model.pth", map_location=device)
    gen.load_state_dict(ckpt["modelG_state_dict"])
    disc.load_state_dict(ckpt["modelD_state_dict"])
    gen.to(device).eval()
    disc.to(device).eval()
    return gen, disc, device

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def init_session_state():
    defaults = {
        "authenticated":  False,
        "auth_user":      "",
        "generated_images":    [],
        "generation_history":  [],
        "scan_history":        [],
        "favorite_seeds":      [],
        "batch_results":       [],
        "total_generated":     0,
        "total_scanned":       0,
        "theme":               "dark",
        "latent_dims":         [0.0]*10,
        "evo_generation":      0,
        "evo_seeds":           [],
        "evo_ratings":         {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ── Auth gate ──────────────────────────────────
if not check_login():
    render_login()
    st.stop()

load_css()

# ─────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────
def postprocess_tensor(t):
    arr = ((t.permute(1,2,0).numpy()*0.5+0.5)*255).clip(0,255).astype("uint8")
    return Image.fromarray(arr)

def apply_adjustments(img, brightness=1.0, contrast=1.0, sharpness=1.0):
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img

def image_to_bytes(img):
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

def gif_to_bytes(frames, fps=10):
    buf = io.BytesIO()
    imageio.mimsave(buf, [np.array(f) for f in frames], format="GIF", fps=fps)
    return buf.getvalue()

def log_generation(seed, ts):
    st.session_state.generation_history.append({"seed": seed, "timestamp": ts})
    st.session_state.total_generated += 1

def log_scan(filename, score, verdict, ts):
    st.session_state.scan_history.append(
        {"file": filename, "score": round(score,4), "verdict": verdict, "timestamp": ts})
    st.session_state.total_scanned += 1

def slerp(val, low, high):
    ln = low  / (torch.norm(low,  dim=1, keepdim=True) + 1e-8)
    hn = high / (torch.norm(high, dim=1, keepdim=True) + 1e-8)
    omega = torch.acos((ln*hn).sum(1).clamp(-1,1))
    so    = torch.sin(omega)
    if so.item() < 1e-8:
        return (1-val)*low + val*high
    return (torch.sin((1-val)*omega)/so).unsqueeze(1)*low + \
           (torch.sin(val*omega)/so).unsqueeze(1)*high

def generate_image(seed, device, gen_model):
    torch.manual_seed(int(seed))
    z = torch.randn(1,100,1,1, device=device)
    with torch.no_grad():
        t = gen_model(z).detach().cpu()[0]
    return postprocess_tensor(t)

TRANSFORM_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
try:
    gen, disc, device = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error  = str(e)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("AnimeGen Pro AI")
st.sidebar.caption(f"Signed in as `{st.session_state.auth_user}`")

theme_label = "☀ Light Mode" if get_theme()=="dark" else "🌑 Dark Mode"
if st.sidebar.button(theme_label, use_container_width=True):
    st.session_state.theme = "light" if get_theme()=="dark" else "dark"
    st.rerun()

if st.sidebar.button("⎋  Sign Out", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.auth_user = ""
    st.rerun()

st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "AI Generator",
    "Vision Scanner",
    "Batch Generator",
    "Evolution Lab",
    "GAN Inversion",
    "Style Transfer",
    "GIF Export",
    "Feature Map Viewer",
    "Image Gallery",
    "Generation History",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Session Stats**")
st.sidebar.metric("Generated", st.session_state.total_generated)
st.sidebar.metric("Scanned",   st.session_state.total_scanned)
st.sidebar.metric("Favorites", len(st.session_state.favorite_seeds))
st.sidebar.markdown(f"**Device:** `{'CUDA' if torch.cuda.is_available() else 'CPU'}`")
st.sidebar.markdown(f"**Model:**  `{'Loaded ✓' if model_loaded else 'Not Found ✗'}`")

st.sidebar.markdown("---")
st.sidebar.markdown("**Export Session**")
export_fmt = st.sidebar.selectbox("Format", ["JSON","CSV"], key="export_fmt")

def build_export():
    data = {
        "exported_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generation_history": st.session_state.generation_history,
        "scan_history":       st.session_state.scan_history,
        "favorite_seeds":     st.session_state.favorite_seeds,
        "total_generated":    st.session_state.total_generated,
        "total_scanned":      st.session_state.total_scanned,
    }
    if export_fmt == "JSON":
        return json.dumps(data, indent=2).encode(), "session_report.json", "application/json"
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["section","seed_or_file","score","verdict","timestamp"])
    for e in st.session_state.generation_history:
        w.writerow(["generation", e["seed"], "", "", e["timestamp"]])
    for e in st.session_state.scan_history:
        w.writerow(["scan", e["file"], e["score"], e["verdict"], e["timestamp"]])
    return buf.getvalue().encode(), "session_report.csv", "text/csv"

exp_data, exp_name, exp_mime = build_export()
st.sidebar.download_button("⬇ Export Report", data=exp_data, file_name=exp_name,
                           mime=exp_mime, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# AI GENERATOR
# ═══════════════════════════════════════════════════════════
if page == "AI Generator":
    st.header("Image Synthesis")
    if not model_loaded: st.error(f"Model error: {model_error}"); st.stop()

    gen_mode = st.radio("Mode", ["Standard","Interpolation","Noise Grid","Latent Sliders"], horizontal=True)
    st.markdown("---")

    if gen_mode == "Standard":
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Controls")
            seed       = st.number_input("Seed", value=42, min_value=0, max_value=999999)
            use_random = st.checkbox("Random seed each run")
            brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
            contrast   = st.slider("Contrast",   0.5, 2.0, 1.0, 0.05)
            sharpness  = st.slider("Sharpness",  0.0, 3.0, 1.0, 0.10)
            btn        = st.button("Generate", use_container_width=True)
            if st.session_state.favorite_seeds:
                chosen = st.selectbox("Load favourite", st.session_state.favorite_seeds)
                if st.button("Load Seed"): seed = chosen
        with c2:
            st.subheader("Output")
            if btn:
                actual = int(time.time()*1000)%999999 if use_random else int(seed)
                img    = apply_adjustments(generate_image(actual, device, gen), brightness, contrast, sharpness)
                img_up = img.resize((256,256), Image.NEAREST)
                st.image(img_up, caption=f"Seed: {actual}", width=256)
                ca, cb = st.columns(2)
                ca.download_button("⬇ PNG", image_to_bytes(img_up), f"anime_{actual}.png", "image/png", use_container_width=True)
                if cb.button("★ Favourite", use_container_width=True):
                    if actual not in st.session_state.favorite_seeds:
                        st.session_state.favorite_seeds.append(actual); st.success("Saved!")
                log_generation(actual, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                st.session_state.generated_images.append((actual, image_to_bytes(img_up)))

    elif gen_mode == "Interpolation":
        st.subheader("Seed Interpolation")
        ca,cb,cc = st.columns(3)
        seed_a = ca.number_input("Seed A", value=42,  min_value=0, max_value=999999)
        seed_b = cb.number_input("Seed B", value=137, min_value=0, max_value=999999)
        steps  = cc.slider("Steps", 3, 12, 7)
        if st.button("Interpolate", use_container_width=True):
            torch.manual_seed(int(seed_a)); z_a = torch.randn(1,100, device=device)
            torch.manual_seed(int(seed_b)); z_b = torch.randn(1,100, device=device)
            cols = st.columns(steps)
            with st.spinner("Interpolating..."):
                for i,col in enumerate(cols):
                    alpha = i/(steps-1)
                    z = slerp(alpha, z_a, z_b).view(1,100,1,1)
                    with torch.no_grad(): t = gen(z).detach().cpu()[0]
                    col.image(postprocess_tensor(t).resize((128,128),Image.NEAREST),
                              caption=f"α={alpha:.2f}", use_column_width=True)

    elif gen_mode == "Noise Grid":
        st.subheader("Noise Exploration Grid")
        base_seed   = st.number_input("Base Seed", value=42, min_value=0, max_value=999999)
        noise_scale = st.slider("Perturbation Scale", 0.01, 0.5, 0.1, 0.01)
        if st.button("Generate Grid", use_container_width=True):
            torch.manual_seed(int(base_seed))
            z_base = torch.randn(1,100,1,1, device=device)
            cols = st.columns(3)
            with st.spinner("Rendering..."):
                for i in range(9):
                    z = z_base if i==0 else z_base + torch.randn_like(z_base)*noise_scale
                    with torch.no_grad(): t = gen(z).detach().cpu()[0]
                    cols[i%3].image(postprocess_tensor(t).resize((128,128),Image.NEAREST),
                                    caption="Base" if i==0 else f"Var {i}", use_column_width=True)

    elif gen_mode == "Latent Sliders":
        st.subheader("Latent Vector Sliders")
        cs, cp = st.columns(2)
        with cs:
            base_seed = st.number_input("Base Seed", value=42, min_value=0, max_value=999999)
            dims = [st.slider(f"Dim {d}", -3.0, 3.0, float(st.session_state.latent_dims[d]), 0.05,
                              key=f"ld_{d}") for d in range(10)]
            st.session_state.latent_dims = dims
            btn_lat = st.button("Generate", use_container_width=True)
        with cp:
            st.subheader("Preview")
            if btn_lat:
                torch.manual_seed(int(base_seed))
                z = torch.randn(100, device=device)
                for d,v in enumerate(dims): z[d] = v
                with torch.no_grad(): t = gen(z.view(1,100,1,1)).detach().cpu()[0]
                img = postprocess_tensor(t).resize((256,256), Image.NEAREST)
                st.image(img, width=256)
                st.download_button("⬇ PNG", image_to_bytes(img), "latent_custom.png", "image/png", use_container_width=True)
                log_generation(f"latent_{base_seed}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                st.session_state.generated_images.append((f"custom_{base_seed}", image_to_bytes(img)))


# ═══════════════════════════════════════════════════════════
# VISION SCANNER
# ═══════════════════════════════════════════════════════════
elif page == "Vision Scanner":
    st.header("Vision Scanner — AI Image Authentication")
    if not model_loaded: st.error(f"Model error: {model_error}"); st.stop()

    uploaded_files = st.file_uploader("Upload Images (multi-select supported)",
                                      type=["jpg","png","jpeg"], accept_multiple_files=True)

    def confidence_bars(score):
        real_pct = score*100; ai_pct = (1-score)*100
        return f"""
        <div style="margin-top:0.5rem;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span style="width:80px;font-size:0.75rem;font-family:monospace;color:#a1a1aa;">REAL</span>
            <div style="flex:1;background:#27272a;border-radius:4px;height:16px;overflow:hidden;">
              <div style="width:{real_pct:.1f}%;background:linear-gradient(90deg,#22c55e,#4ade80);height:100%;border-radius:4px;"></div>
            </div>
            <span style="width:42px;font-size:0.75rem;font-family:monospace;color:#e4e4e7;">{real_pct:.1f}%</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="width:80px;font-size:0.75rem;font-family:monospace;color:#a1a1aa;">AI-GEN</span>
            <div style="flex:1;background:#27272a;border-radius:4px;height:16px;overflow:hidden;">
              <div style="width:{ai_pct:.1f}%;background:linear-gradient(90deg,#ef4444,#f87171);height:100%;border-radius:4px;"></div>
            </div>
            <span style="width:42px;font-size:0.75rem;font-family:monospace;color:#e4e4e7;">{ai_pct:.1f}%</span>
          </div>
        </div>"""

    if uploaded_files:
        if len(uploaded_files) == 1:
            uf  = uploaded_files[0]
            img = Image.open(uf).convert("RGB").resize((64,64))
            ci, cr = st.columns(2)
            ci.image(img.resize((200,200),Image.NEAREST), caption="Input", width=200)
            ci.download_button("⬇ Resized", image_to_bytes(img), f"scanned_{uf.name}", "image/png")
            with cr:
                if st.button("Analyze", use_container_width=True):
                    t = TRANSFORM_NORM(img).unsqueeze(0).to(device)
                    with torch.no_grad(): score = disc(t).item()
                    verdict = "REAL" if score>0.5 else "AI-GENERATED"
                    conf    = score if score>0.5 else 1-score
                    st.markdown(f"**Score:** `{score:.4f}`  **Verdict:** `{verdict}`  **Conf:** `{conf*100:.1f}%`")
                    st.progress(float(score))
                    st.markdown(confidence_bars(score), unsafe_allow_html=True)
                    if verdict=="REAL": st.success("Classified as REAL.")
                    else:               st.error("Classified as AI-GENERATED.")
                    log_scan(uf.name, score, verdict, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        else:
            st.subheader(f"Batch Scan — {len(uploaded_files)} images")
            if st.button("Analyze All", use_container_width=True):
                results, scores = [], []
                with st.spinner("Analyzing..."):
                    for uf in uploaded_files:
                        img = Image.open(uf).convert("RGB").resize((64,64))
                        t   = TRANSFORM_NORM(img).unsqueeze(0).to(device)
                        with torch.no_grad(): score = disc(t).item()
                        verdict = "REAL" if score>0.5 else "AI-GENERATED"
                        conf    = score if score>0.5 else 1-score
                        results.append({"filename":uf.name,"score":round(score,4),
                                        "verdict":verdict,"confidence":f"{conf*100:.1f}%"})
                        scores.append(score)
                        log_scan(uf.name, score, verdict, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                h0,h1,h2,h3 = st.columns([3,2,2,2])
                for h,lbl in zip([h0,h1,h2,h3],["**Filename**","**Score**","**Verdict**","**Confidence**"]):
                    h.markdown(lbl)
                st.markdown("---")
                for r in results:
                    r0,r1,r2,r3 = st.columns([3,2,2,2])
                    clr = "#22c55e" if r["verdict"]=="REAL" else "#ef4444"
                    r0.write(r["filename"]); r1.write(f'`{r["score"]}`')
                    r2.markdown(f'<span style="color:{clr};font-weight:600;">{r["verdict"]}</span>', unsafe_allow_html=True)
                    r3.write(r["confidence"])

                # Score histogram
                st.markdown("---"); st.subheader("Score Distribution")
                bin_edges = np.linspace(0,1,11); counts,_ = np.histogram(scores, bins=bin_edges)
                mx = max(counts) if max(counts)>0 else 1
                hist = '<div style="display:flex;align-items:flex-end;gap:4px;height:120px;padding:8px;background:#18181b;border-radius:8px;border:1px solid #27272a;">'
                for i,cnt in enumerate(counts):
                    clr = "#22c55e" if bin_edges[i]>=0.5 else "#ef4444"
                    pct = int(cnt/mx*100)
                    hist += (f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:2px;">'
                             f'<span style="font-size:0.6rem;color:#71717a;">{cnt}</span>'
                             f'<div style="width:100%;height:{pct}%;background:{clr};border-radius:3px 3px 0 0;min-height:2px;"></div>'
                             f'<span style="font-size:0.55rem;color:#52525b;">{bin_edges[i]:.1f}</span></div>')
                hist += '</div>'
                st.markdown(hist, unsafe_allow_html=True)

                # Comparison
                st.markdown("---"); st.subheader("Side-by-Side Comparison")
                fnames = [r["filename"] for r in results]
                cs1,cs2 = st.columns(2)
                sel1 = cs1.selectbox("Image A", fnames, key="cmp1")
                sel2 = cs2.selectbox("Image B", fnames, index=min(1,len(fnames)-1), key="cmp2")
                cc1,cc2 = st.columns(2)
                for uf in uploaded_files:
                    if uf.name==sel1:
                        ra = next(r for r in results if r["filename"]==sel1)
                        cc1.image(Image.open(uf).convert("RGB").resize((64,64)).resize((180,180),Image.NEAREST),
                                  caption=f"{sel1} | {ra['verdict']} ({ra['score']})", width=180)
                    if uf.name==sel2:
                        rb = next(r for r in results if r["filename"]==sel2)
                        cc2.image(Image.open(uf).convert("RGB").resize((64,64)).resize((180,180),Image.NEAREST),
                                  caption=f"{sel2} | {rb['verdict']} ({rb['score']})", width=180)


# ═══════════════════════════════════════════════════════════
# BATCH GENERATOR
# ═══════════════════════════════════════════════════════════
elif page == "Batch Generator":
    st.header("Batch Image Generation")
    if not model_loaded: st.error(f"Model error: {model_error}"); st.stop()

    num_images = st.slider("Number of Images", 1, 16, 4)
    base_seed  = st.number_input("Base Seed", value=0, min_value=0, step=1)

    if st.button("Generate Batch", use_container_width=True):
        cols, batch = st.columns(4), []
        with st.spinner("Generating..."):
            for i in range(num_images):
                s   = int(base_seed)+i
                img = generate_image(s, device, gen).resize((128,128), Image.NEAREST)
                batch.append((s, img))
                cols[i%4].image(img, caption=f"Seed {s}", use_column_width=True)
                st.session_state.generated_images.append((s, image_to_bytes(img)))
        st.session_state.batch_results    = batch
        st.session_state.total_generated += num_images
        st.success(f"Generated {num_images} images.")
        st.markdown("---"); st.subheader("Downloads")
        dcols = st.columns(4)
        for idx,(s,img) in enumerate(batch):
            dcols[idx%4].download_button(f"Seed {s}", image_to_bytes(img), f"batch_{s}.png","image/png")


# ═══════════════════════════════════════════════════════════
# EVOLUTION LAB  — Feature 1: Genetic Algorithm
# ═══════════════════════════════════════════════════════════
elif page == "Evolution Lab":
    st.header("Evolution Lab — Seed Genetic Algorithm")
    st.write(
        "Rate 4 generated faces each round. Seeds you mark **Good** breed the next generation "
        "through latent-space crossover and mutation."
    )
    if not model_loaded: st.error(f"Model error: {model_error}"); st.stop()

    def reset_evolution():
        st.session_state.evo_generation = 1
        st.session_state.evo_seeds      = [random.randint(0,999999) for _ in range(4)]
        st.session_state.evo_ratings    = {}

    if st.session_state.evo_generation == 0:
        reset_evolution()

    if st.button("↺ Restart Evolution"):
        reset_evolution(); st.rerun()

    st.markdown(f"**Generation #{st.session_state.evo_generation}**")
    st.markdown("---")

    seeds   = st.session_state.evo_seeds
    ratings = st.session_state.evo_ratings
    cols    = st.columns(4)

    for i, seed in enumerate(seeds):
        with cols[i]:
            img = generate_image(seed, device, gen).resize((128,128), Image.NEAREST)
            st.image(img, caption=f"Seed {seed}", use_column_width=True)
            liked = st.checkbox("👍 Good",
                                key=f"evo_{st.session_state.evo_generation}_{i}",
                                value=ratings.get(i, False))
            ratings[i] = liked
    st.session_state.evo_ratings = ratings

    if st.button("Breed Next Generation ➜", use_container_width=True):
        good = [seeds[i] for i,v in ratings.items() if v]
        if not good: st.warning("Mark at least one face as Good."); st.stop()

        new_seeds = []
        while len(new_seeds) < 4:
            p1, p2 = random.choice(good), random.choice(good)
            torch.manual_seed(p1); z1 = torch.randn(100)
            torch.manual_seed(p2); z2 = torch.randn(100)
            cut     = random.randint(10, 90)
            child_z = torch.cat([z1[:cut], z2[cut:]])
            for idx in random.sample(range(100), random.randint(3,15)):
                child_z[idx] += random.gauss(0, 0.3)
            new_seeds.append(int(abs(child_z.sum().item()) * 1e5) % 999999)

        st.session_state.evo_seeds      = new_seeds
        st.session_state.evo_generation += 1
        st.session_state.evo_ratings    = {}
        st.rerun()


# ═══════════════════════════════════════════════════════════
# GAN INVERSION  — Feature 2
# ═══════════════════════════════════════════════════════════
elif page == "GAN Inversion":
    st.header("GAN Inversion — Image → Latent Vector")
    st.write("Upload an anime face. An optimization loop finds the latent vector that best reconstructs it.")
    if not model_loaded: st.error(f"Model error: {model_error}"); st.stop()

    uf = st.file_uploader("Upload Anime Face", type=["jpg","png","jpeg"])
    if uf:
        target_img = Image.open(uf).convert("RGB").resize((64,64))
        target_t   = TRANSFORM_NORM(target_img).unsqueeze(0).to(device)

        c1, c2 = st.columns(2)
        c1.subheader("Target")
        c1.image(target_img.resize((160,160),Image.NEAREST), width=160)

        n_steps = st.slider("Optimization Steps", 50, 500, 150, 50)
        lr_val  = st.select_slider("Learning Rate", [0.001,0.005,0.01,0.05,0.1], value=0.01)

        if st.button("Run Inversion", use_container_width=True):
            z_opt = torch.randn(1,100,1,1, device=device, requires_grad=True)
            opt   = optim.Adam([z_opt], lr=float(lr_val))
            mse   = nn.MSELoss()
            prog  = st.progress(0); status = st.empty()

            for step in range(n_steps):
                opt.zero_grad()
                loss = mse(gen(z_opt), target_t)
                loss.backward(); opt.step()
                if (step+1) % 10 == 0:
                    prog.progress((step+1)/n_steps)
                    status.caption(f"Step {step+1}/{n_steps}  Loss: {loss.item():.5f}")

            with torch.no_grad(): t = gen(z_opt).detach().cpu()[0]
            recon = postprocess_tensor(t).resize((160,160), Image.NEAREST)
            c2.subheader("Reconstruction")
            c2.image(recon, caption=f"Loss: {loss.item():.5f}", width=160)
            c2.download_button("⬇ Download", image_to_bytes(recon), "inversion.png","image/png", use_container_width=True)

            z_flat = z_opt.detach().cpu().view(100)
            st.session_state.latent_dims = [float(z_flat[d]) for d in range(10)]
            st.info("Latent dims 0-9 saved → go to **AI Generator → Latent Sliders** to fine-tune.")
            log_generation("inversion", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            st.session_state.generated_images.append(("inversion", image_to_bytes(recon)))


# ═══════════════════════════════════════════════════════════
# STYLE TRANSFER  — Feature 3
# ═══════════════════════════════════════════════════════════
elif page == "Style Transfer":
    st.header("Style Transfer — Discriminator Feature Matching")
    st.write("Upload a **content** and a **style** image. The output matches content structure with style texture.")
    if not model_loaded: st.error(f"Model error: {model_error}"); st.stop()

    c_col, s_col = st.columns(2)
    cf = c_col.file_uploader("Content Image", type=["jpg","png","jpeg"], key="st_c")
    sf = s_col.file_uploader("Style Image",   type=["jpg","png","jpeg"], key="st_s")

    n_steps_st = st.slider("Steps",          50, 300, 100, 50)
    cw         = st.slider("Content Weight", 0.1, 10.0, 1.0, 0.1)
    sw         = st.slider("Style Weight",   0.1, 10.0, 5.0, 0.1)

    if cf and sf:
        ci = Image.open(cf).convert("RGB").resize((64,64))
        si = Image.open(sf).convert("RGB").resize((64,64))
        p1,p2 = st.columns(2)
        p1.image(ci.resize((128,128),Image.NEAREST), caption="Content", width=128)
        p2.image(si.resize((128,128),Image.NEAREST), caption="Style",   width=128)

        if st.button("Run Style Transfer", use_container_width=True):
            ct = TRANSFORM_NORM(ci).unsqueeze(0).to(device)
            st_ = TRANSFORM_NORM(si).unsqueeze(0).to(device)

            def gram(x):
                b,c,h,w = x.shape; f = x.view(b,c,h*w)
                return torch.bmm(f, f.transpose(1,2)) / (c*h*w)

            with torch.no_grad():
                s_maps = disc.get_feature_maps(st_)
                c_maps = disc.get_feature_maps(ct)

            style_grams   = [gram(m.to(device)) for m in s_maps]
            content_feats = [m.to(device) for m in c_maps]

            z_opt = torch.randn(1,100,1,1, device=device, requires_grad=True)
            opt   = optim.Adam([z_opt], lr=0.01); mse = nn.MSELoss()
            prog  = st.progress(0); status = st.empty()

            for step in range(n_steps_st):
                opt.zero_grad()
                out_maps = disc.get_feature_maps(gen(z_opt))
                c_loss = sum(mse(out_maps[i].to(device), content_feats[i]) for i in range(len(out_maps))) * cw
                s_loss = sum(mse(gram(out_maps[i].to(device)), style_grams[i]) for i in range(len(out_maps))) * sw
                loss   = c_loss + s_loss
                loss.backward(); opt.step()
                if (step+1) % 10 == 0:
                    prog.progress((step+1)/n_steps_st)
                    status.caption(f"Step {step+1}/{n_steps_st}  Loss: {loss.item():.4f}")

            with torch.no_grad(): t = gen(z_opt).detach().cpu()[0]
            result = postprocess_tensor(t).resize((256,256), Image.NEAREST)
            st.subheader("Result")
            st.image(result, width=256)
            st.download_button("⬇ Download", image_to_bytes(result), "style_transfer.png","image/png", use_container_width=True)
            log_generation("style_transfer", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            st.session_state.generated_images.append(("style_tf", image_to_bytes(result)))


# ═══════════════════════════════════════════════════════════
# GIF EXPORT  — Feature 4
# ═══════════════════════════════════════════════════════════
elif page == "GIF Export":
    st.header("GIF Export — Animated Interpolation")
    st.write("Generate a downloadable animated GIF transitioning between two seed faces.")
    if not model_loaded: st.error(f"Model error: {model_error}"); st.stop()

    cg1,cg2,cg3 = st.columns(3)
    seed_a   = cg1.number_input("Seed A", value=42,  min_value=0, max_value=999999)
    seed_b   = cg2.number_input("Seed B", value=137, min_value=0, max_value=999999)
    n_frames = cg3.slider("Frames", 8, 48, 16)
    fps_val  = st.slider("FPS", 4, 24, 10)
    img_sz   = st.select_slider("Frame Size (px)", [64,128,256], value=128)
    pingpong = st.checkbox("Ping-pong loop (A → B → A)", value=True)

    if st.button("Generate GIF", use_container_width=True):
        torch.manual_seed(int(seed_a)); z_a = torch.randn(1,100, device=device)
        torch.manual_seed(int(seed_b)); z_b = torch.randn(1,100, device=device)
        alphas = list(np.linspace(0,1,n_frames))
        if pingpong: alphas = alphas + alphas[::-1]

        frame_list, prog = [], st.progress(0)
        with st.spinner("Rendering frames..."):
            for i,alpha in enumerate(alphas):
                z = slerp(alpha, z_a, z_b).view(1,100,1,1)
                with torch.no_grad(): t = gen(z).detach().cpu()[0]
                frame_list.append(postprocess_tensor(t).resize((img_sz,img_sz), Image.NEAREST))
                prog.progress((i+1)/len(alphas))

        gif_bytes = gif_to_bytes(frame_list, fps=fps_val)
        st.success(f"{len(frame_list)} frames @ {fps_val} fps")
        pc1,pc2 = st.columns(2)
        pc1.image(frame_list[0],  caption="First frame", width=img_sz)
        pc2.image(frame_list[-1], caption="Last frame",  width=img_sz)
        st.download_button("⬇ Download GIF", data=gif_bytes,
                           file_name=f"interp_{seed_a}_{seed_b}.gif",
                           mime="image/gif", use_container_width=True)


# ═══════════════════════════════════════════════════════════
# FEATURE MAP VIEWER  — Feature 5
# ═══════════════════════════════════════════════════════════
elif page == "Feature Map Viewer":
    st.header("Discriminator Feature Map Visualizer")
    st.write("Inspect what each discriminator layer activates on — averaged heatmap + individual channels.")
    if not model_loaded: st.error(f"Model error: {model_error}"); st.stop()

    uf_fm = st.file_uploader("Upload Image", type=["jpg","png","jpeg"], key="fm_up")
    layer_names = [
        "Layer 1 — Low-level Edges",
        "Layer 2 — Texture Patterns",
        "Layer 3 — Mid-level Structures",
        "Layer 4 — Semantic Features",
    ]

    if uf_fm:
        img_fm = Image.open(uf_fm).convert("RGB").resize((64,64))
        t_fm   = TRANSFORM_NORM(img_fm).unsqueeze(0).to(device)
        ci, cs = st.columns(2)
        ci.subheader("Input")
        ci.image(img_fm.resize((160,160),Image.NEAREST), width=160)

        with torch.no_grad():
            score_fm = disc(t_fm).item()
            fmaps    = disc.get_feature_maps(t_fm)

        verdict_fm = "REAL" if score_fm>0.5 else "AI-GENERATED"
        cs.subheader("Classification")
        cs.metric("Score",   f"{score_fm:.4f}")
        cs.metric("Verdict", verdict_fm)

        st.markdown("---"); st.subheader("Activation Heatmaps per Layer")

        for li,(fmap,name) in enumerate(zip(fmaps,layer_names)):
            st.markdown(f"**{name}** — shape `{tuple(fmap.shape)}`")
            heat     = fmap[0].mean(0).numpy()
            heat     = (heat-heat.min())/(heat.max()-heat.min()+1e-8)
            heat_u8  = (heat*255).astype(np.uint8)

            # Purple-tinted mean heatmap
            heat_rgb = np.stack([
                (heat_u8*0.4).astype(np.uint8),
                (heat_u8*0.2).astype(np.uint8),
                heat_u8
            ], axis=-1)
            heat_rgb_img = Image.fromarray(heat_rgb).resize((64,64), Image.NEAREST)

            n_show   = min(8, fmap.shape[1])
            ch_cols  = st.columns(n_show+1)
            ch_cols[0].image(heat_rgb_img, caption="Avg", use_column_width=True)

            for ci_idx in range(n_show):
                ch = fmap[0,ci_idx].numpy()
                ch = (ch-ch.min())/(ch.max()-ch.min()+1e-8)
                ch_img = Image.fromarray((ch*255).astype(np.uint8), mode="L").resize((64,64),Image.NEAREST)
                ch_cols[ci_idx+1].image(ch_img, caption=f"Ch {ci_idx}", use_column_width=True)
            st.markdown("---")


# ═══════════════════════════════════════════════════════════
# IMAGE GALLERY
# ═══════════════════════════════════════════════════════════
elif page == "Image Gallery":
    st.header("Image Gallery")
    if not st.session_state.generated_images:
        st.info("No images generated yet.")
    else:
        total = len(st.session_state.generated_images)
        st.caption(f"{total} image{'s' if total!=1 else ''} in session")
        if st.button("🗑 Clear Gallery"):
            st.session_state.generated_images = []; st.rerun()
        cols = st.columns(4)
        for i,(seed_lbl,img_bytes) in enumerate(reversed(st.session_state.generated_images)):
            col = cols[i%4]
            col.image(Image.open(io.BytesIO(img_bytes)), caption=f"Seed: {seed_lbl}", use_column_width=True)
            col.download_button("⬇", img_bytes, f"gallery_{seed_lbl}.png","image/png",
                                key=f"gal_{i}", use_container_width=True)


# ═══════════════════════════════════════════════════════════
# GENERATION HISTORY
# ═══════════════════════════════════════════════════════════
elif page == "Generation History":
    st.header("Session History")
    tab_gen, tab_scan = st.tabs(["Generation Log","Scan Log"])

    with tab_gen:
        st.subheader("Image Generation History")
        if st.session_state.generation_history:
            for e in reversed(st.session_state.generation_history):
                st.markdown(f"- Seed `{e['seed']}` at `{e['timestamp']}`")
        else:
            st.info("No images generated yet.")
        if st.button("Clear Generation History"):
            st.session_state.generation_history = []; st.success("Cleared.")

    with tab_scan:
        st.subheader("Image Scan History")
        if st.session_state.scan_history:
            for e in reversed(st.session_state.scan_history):
                clr = "#22c55e" if e["verdict"]=="REAL" else "#ef4444"
                st.markdown(
                    f"- `{e['file']}` — `{e['score']}` — "
                    f'<span style="color:{clr};font-weight:600;">{e["verdict"]}</span>'
                    f" — `{e['timestamp']}`", unsafe_allow_html=True)
        else:
            st.info("No scans yet.")
        if st.button("Clear Scan History"):
            st.session_state.scan_history = []; st.success("Cleared.")
