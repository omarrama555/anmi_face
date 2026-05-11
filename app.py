import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import time
import json
import csv
import os
import hashlib
import imageio
from datetime import datetime

# --- Page Config ---
st.set_page_config(
    page_title="AniVision AI",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# LOGIN SYSTEM
# ══════════════════════════════════════════════════════════════
USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "user":  hashlib.sha256("user123".encode()).hexdigest(),
}

def check_login(username, password):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return USERS.get(username) == hashed

def show_login_page():
    st.markdown("""
    <style>
    .login-box {
        max-width: 420px;
        margin: 80px auto 0 auto;
        padding: 2.5rem 2rem;
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 16px;
        box-shadow: 0 8px 40px rgba(99,102,241,0.15);
    }
    .login-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        color: #6366f1;
        margin-bottom: 0.25rem;
        font-family: Arial, sans-serif;
    }
    .login-sub {
        text-align: center;
        color: #71717a;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        font-family: Arial, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown('<div class="login-title">AnimeGen Pro AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">GAN-based Anime Generation & Authentication</div>', unsafe_allow_html=True)
        st.markdown("---")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        if st.button("Login", use_container_width=True):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password.")
        

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    show_login_page()
    st.stop()


# ══════════════════════════════════════════════════════════════
# THEME & CSS
# ══════════════════════════════════════════════════════════════
def get_theme():
    return st.session_state.get("theme", "dark")

def load_css():
    theme = get_theme()
    try:
        with open("style.css") as f:
            css = f.read()
    except FileNotFoundError:
        css = ""
    if theme == "light":
        light_overrides = """
        .stApp {
            background-color: #f4f4f5 !important;
            background-image: radial-gradient(ellipse 80% 40% at 50% -10%, rgba(99,102,241,0.08), transparent) !important;
            color: #18181b !important;
        }
        h1 { color: #18181b !important; border-bottom-color: #d4d4d8 !important; }
        h2, h3 { color: #52525b !important; }
        section[data-testid="stSidebar"] { background-color: #e4e4e7 !important; border-right-color: #d4d4d8 !important; }
        .stButton > button { background: #e4e4e7 !important; border-color: #a1a1aa !important; color: #18181b !important; }
        .stButton > button:hover { background: #6366f1 !important; border-color: #6366f1 !important; color: #fff !important; }
        .stNumberInput input, .stTextInput input, .stSelectbox select { background-color: #ffffff !important; border-color: #a1a1aa !important; color: #18181b !important; }
        [data-testid="stMetricValue"] { color: #18181b !important; }
        code { background-color: #e4e4e7 !important; border-color: #d4d4d8 !important; color: #6366f1 !important; }
        [data-testid="stFileUploader"] { background-color: #ffffff !important; border-color: #a1a1aa !important; }
        """
        css += light_overrides
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

    def get_feature_maps(self, x):
        """Return intermediate feature maps from each conv block."""
        features = []
        layer_names = []
        # Run through each named sub-module manually
        blocks = [
            ("Block 1 — Conv(3→64)", [0, 1]),
            ("Block 2 — Conv(64→128) + BN", [2, 3, 4]),
            ("Block 3 — Conv(128→256) + BN", [5, 6, 7]),
            ("Block 4 — Conv(256→512) + BN", [8, 9, 10]),
        ]
        out = x
        for name, indices in blocks:
            for idx in indices:
                out = self.main[idx](out)
            features.append(out.detach().cpu())
            layer_names.append(name)
        return features, layer_names


def _remap_discriminator_state_dict(state_dict):
    if any(k.startswith("main.") for k in state_dict.keys()):
        return state_dict
    remapped = {}
    layer_map = {
        "layers.0.0.": "main.0.",
        "layers.1.0.": "main.2.",
        "layers.1.1.": "main.3.",
        "layers.2.0.": "main.5.",
        "layers.2.1.": "main.6.",
        "layers.3.0.": "main.8.",
        "layers.3.1.": "main.9.",
        "final.0.":    "main.11.",
    }
    for old_key, tensor in state_dict.items():
        new_key = old_key
        for old_prefix, new_prefix in layer_map.items():
            if old_key.startswith(old_prefix):
                new_key = new_prefix + old_key[len(old_prefix):]
                break
        remapped[new_key] = tensor
    return remapped


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = Generator()
    disc = Discriminator()
    checkpoint = torch.load("best_gan_model.pth", map_location=device)
    gen.load_state_dict(checkpoint["modelG_state_dict"])
    disc_state = _remap_discriminator_state_dict(checkpoint["modelD_state_dict"])
    disc.load_state_dict(disc_state)
    gen.to(device).eval()
    disc.to(device).eval()
    return gen, disc, device


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "generated_images": [],
        "generation_history": [],
        "scan_history": [],
        "favorite_seeds": [],
        "batch_results": [],
        "total_generated": 0,
        "total_scanned": 0,
        "theme": "dark",
        "latent_dims": [0.0] * 10,
        # Evolution / GA
        "evo_population": [],
        "evo_generation": 0,
        # GAN Inversion
        "inverted_latent": None,
        "inverted_image_bytes": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()
load_css()


# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════
def postprocess_tensor(img_tensor):
    arr = ((img_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)

def apply_image_adjustments(img, brightness=1.0, contrast=1.0, sharpness=1.0):
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img

def image_to_bytes(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def log_generation(seed, timestamp):
    st.session_state.generation_history.append({"seed": seed, "timestamp": timestamp})
    st.session_state.total_generated += 1

def log_scan(filename, score, verdict, timestamp):
    st.session_state.scan_history.append({"file": filename, "score": round(score, 4), "verdict": verdict, "timestamp": timestamp})
    st.session_state.total_scanned += 1

def slerp(val, low, high):
    low_norm  = low  / (torch.norm(low,  dim=1, keepdim=True) + 1e-8)
    high_norm = high / (torch.norm(high, dim=1, keepdim=True) + 1e-8)
    omega = torch.acos((low_norm * high_norm).sum(1).clamp(-1, 1))
    so = torch.sin(omega)
    if so.item() < 1e-8:
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + \
           (torch.sin(val * omega) / so).unsqueeze(1) * high

def seed_to_latent(seed, device):
    torch.manual_seed(int(seed))
    return torch.randn(1, 100, device=device)

def generate_image_from_z(z, gen, device):
    z4 = z.view(1, 100, 1, 1).to(device)
    with torch.no_grad():
        return gen(z4).detach().cpu()[0]

img_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# ══════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════
try:
    gen, disc, device = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.title("AnimeGen Pro AI")
st.sidebar.caption(f"Logged in as: **{st.session_state.username}**")

theme_label = "☀ Light Mode" if get_theme() == "dark" else "🌑 Dark Mode"
if st.sidebar.button(theme_label, use_container_width=True):
    st.session_state.theme = "light" if get_theme() == "dark" else "dark"
    st.rerun()

if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "AI Generator",
        "Vision Scanner",
        "Batch Generator",
        "Image Gallery",
        "Generation History",
        "── Advanced ──",
        "Seed Evolution (GA)",
        "GAN Inversion",
        "Style Transfer",
        "GIF Export",
        "Feature Map Visualizer",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Session Stats**")
st.sidebar.metric("Images Generated", st.session_state.total_generated)
st.sidebar.metric("Images Scanned",   st.session_state.total_scanned)
st.sidebar.metric("Favorites Saved",  len(st.session_state.favorite_seeds))
st.sidebar.markdown(f"**Device:** `{'CUDA' if torch.cuda.is_available() else 'CPU'}`")
st.sidebar.markdown(f"**Model:** `{'Loaded ✓' if model_loaded else 'Not Found ✗'}`")

st.sidebar.markdown("---")
st.sidebar.markdown("**Export Session**")
export_fmt = st.sidebar.selectbox("Format", ["JSON", "CSV"], key="export_fmt")

def build_export():
    data = {
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generation_history": st.session_state.generation_history,
        "scan_history": st.session_state.scan_history,
        "favorite_seeds": st.session_state.favorite_seeds,
        "total_generated": st.session_state.total_generated,
        "total_scanned": st.session_state.total_scanned,
    }
    if export_fmt == "JSON":
        return json.dumps(data, indent=2).encode(), "session_report.json", "application/json"
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["section", "seed_or_file", "score", "verdict", "timestamp"])
    for e in st.session_state.generation_history:
        writer.writerow(["generation", e["seed"], "", "", e["timestamp"]])
    for e in st.session_state.scan_history:
        writer.writerow(["scan", e["file"], e["score"], e["verdict"], e["timestamp"]])
    return buf.getvalue().encode(), "session_report.csv", "text/csv"

export_data, export_name, export_mime = build_export()
st.sidebar.download_button("⬇ Export Report", data=export_data,
                           file_name=export_name, mime=export_mime,
                           use_container_width=True)


# ══════════════════════════════════════════════════════════════
# DIVIDER PAGE (non-clickable header)
# ══════════════════════════════════════════════════════════════
if page == "── Advanced ──":
    st.info("اختر إحدى الأدوات المتقدمة من القائمة الجانبية.")
    st.stop()


# ══════════════════════════════════════════════════════════════
# PAGE: AI GENERATOR
# ══════════════════════════════════════════════════════════════
if page == "AI Generator":
    st.header("Image Synthesis")

    if not model_loaded:
        st.error(f"Model error: {model_error}")
        st.stop()

    gen_mode = st.radio(
        "Generation Mode",
        ["Standard", "Interpolation", "Noise Grid", "Latent Sliders"],
        horizontal=True
    )
    st.markdown("---")

    # ── Standard ──
    if gen_mode == "Standard":
        col_ctrl, col_out = st.columns([1, 1])
        with col_ctrl:
            st.subheader("Controls")
            seed = st.number_input("Random Seed", value=42, min_value=0, max_value=999999, step=1)
            use_random_seed = st.checkbox("Use random seed on each run", value=False)
            st.markdown("**Post-Processing**")
            brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
            contrast   = st.slider("Contrast",   0.5, 2.0, 1.0, 0.05)
            sharpness  = st.slider("Sharpness",  0.0, 3.0, 1.0, 0.1)
            generate_btn = st.button("Generate Image", use_container_width=True)
            if st.session_state.favorite_seeds:
                st.markdown("**Favorite Seeds**")
                chosen = st.selectbox("Load a favorite seed", options=st.session_state.favorite_seeds)
                if st.button("Load Selected Seed"):
                    seed = chosen
        with col_out:
            st.subheader("Output")
            if generate_btn:
                actual_seed = int(time.time() * 1000) % 999999 if use_random_seed else int(seed)
                torch.manual_seed(actual_seed)
                noise = torch.randn(1, 100, 1, 1, device=device)
                with torch.no_grad():
                    img_tensor = gen(noise).detach().cpu()[0]
                img = postprocess_tensor(img_tensor)
                img = apply_image_adjustments(img, brightness, contrast, sharpness)
                img_up = img.resize((256, 256), Image.NEAREST)
                st.image(img_up, caption=f"Seed: {actual_seed}", use_column_width=False, width=256)
                col_dl, col_fav = st.columns(2)
                col_dl.download_button("⬇ Download PNG", data=image_to_bytes(img_up),
                                       file_name=f"anime_seed_{actual_seed}.png", mime="image/png",
                                       use_container_width=True)
                if col_fav.button("★ Save to Favorites", use_container_width=True):
                    if actual_seed not in st.session_state.favorite_seeds:
                        st.session_state.favorite_seeds.append(actual_seed)
                        st.success(f"Seed {actual_seed} saved.")
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_generation(actual_seed, ts)
                st.session_state.generated_images.append((actual_seed, image_to_bytes(img_up)))

    # ── Interpolation ──
    elif gen_mode == "Interpolation":
        st.subheader("Seed Interpolation")
        st.write("Generate a smooth transition between two different faces.")
        col_a, col_b, col_c = st.columns(3)
        seed_a = col_a.number_input("Seed A", value=42,  min_value=0, max_value=999999, step=1)
        seed_b = col_b.number_input("Seed B", value=137, min_value=0, max_value=999999, step=1)
        steps  = col_c.slider("Interpolation Steps", min_value=3, max_value=12, value=7)
        if st.button("Generate Interpolation", use_container_width=True):
            z_a = seed_to_latent(seed_a, device)
            z_b = seed_to_latent(seed_b, device)
            cols = st.columns(steps)
            with st.spinner("Interpolating latent space..."):
                for i, col in enumerate(cols):
                    alpha = i / (steps - 1)
                    z_interp = slerp(alpha, z_a, z_b).view(1, 100, 1, 1)
                    with torch.no_grad():
                        img_tensor = gen(z_interp).detach().cpu()[0]
                    img = postprocess_tensor(img_tensor).resize((128, 128), Image.NEAREST)
                    col.image(img, caption=f"α={alpha:.2f}", use_column_width=True)
            st.success(f"Generated {steps}-step interpolation from seed {seed_a} → {seed_b}")

    # ── Noise Grid ──
    elif gen_mode == "Noise Grid":
        st.subheader("Noise Exploration Grid")
        st.write("9 variations around a base seed.")
        seed_base   = st.number_input("Base Seed", value=42, min_value=0, max_value=999999, step=1)
        noise_scale = st.slider("Perturbation Scale", 0.01, 0.5, 0.1, 0.01)
        if st.button("Generate Grid", use_container_width=True):
            torch.manual_seed(int(seed_base))
            z_base = torch.randn(1, 100, 1, 1, device=device)
            cols = st.columns(3)
            with st.spinner("Generating 9-image grid..."):
                for i in range(9):
                    if i == 0:
                        z = z_base
                    else:
                        z = z_base + torch.randn_like(z_base) * noise_scale
                    with torch.no_grad():
                        img_tensor = gen(z).detach().cpu()[0]
                    img = postprocess_tensor(img_tensor).resize((128, 128), Image.NEAREST)
                    cols[i % 3].image(img, caption="Base" if i == 0 else f"Var {i}", use_column_width=True)
            st.success("Noise grid complete.")

    # ── Latent Sliders ──
    elif gen_mode == "Latent Sliders":
        st.subheader("Latent Vector Sliders")
        st.write("Control 10 dimensions of the 100-dim noise vector manually.")
        col_sliders, col_preview = st.columns([1, 1])
        with col_sliders:
            base_seed = st.number_input("Base Seed", value=42, min_value=0, max_value=999999, step=1)
            st.markdown("**Manual Dimensions (0–9)**")
            dims = []
            for d in range(10):
                val = st.slider(f"Dim {d}", -3.0, 3.0,
                                float(st.session_state.latent_dims[d]), 0.05,
                                key=f"latent_dim_{d}")
                dims.append(val)
            st.session_state.latent_dims = dims
            generate_latent_btn = st.button("Generate from Sliders", use_container_width=True)
        with col_preview:
            st.subheader("Preview")
            if generate_latent_btn:
                torch.manual_seed(int(base_seed))
                z = torch.randn(100, device=device)
                for d, v in enumerate(dims):
                    z[d] = v
                z = z.view(1, 100, 1, 1)
                with torch.no_grad():
                    img_tensor = gen(z).detach().cpu()[0]
                img = postprocess_tensor(img_tensor).resize((256, 256), Image.NEAREST)
                st.image(img, caption="Custom Latent Vector", use_column_width=False, width=256)
                st.download_button("⬇ Download PNG", data=image_to_bytes(img),
                                   file_name="latent_custom.png", mime="image/png",
                                   use_container_width=True)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_generation(f"latent_custom_{base_seed}", ts)
                st.session_state.generated_images.append((f"custom_{base_seed}", image_to_bytes(img)))


# ══════════════════════════════════════════════════════════════
# PAGE: VISION SCANNER
# ══════════════════════════════════════════════════════════════
elif page == "Vision Scanner":
    st.header("Vision Scanner — AI Image Authentication")

    if not model_loaded:
        st.error(f"Model error: {model_error}")
        st.stop()

    st.write("Upload anime face images. Score > 0.5 → Real, < 0.5 → AI-Generated.")
    uploaded_files = st.file_uploader(
        "Upload Images (JPG / PNG — multi-select supported)",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) == 1:
            uploaded_file = uploaded_files[0]
            img = Image.open(uploaded_file).convert("RGB").resize((64, 64))
            col_img, col_res = st.columns([1, 1])
            with col_img:
                st.image(img.resize((200, 200), Image.NEAREST), caption="Input Image", width=200)
                st.download_button("⬇ Download Resized", data=image_to_bytes(img),
                                   file_name=f"scanned_{uploaded_file.name}", mime="image/png")
            with col_res:
                if st.button("Analyze Image", use_container_width=True):
                    img_tensor = img_transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = disc(img_tensor).item()
                    verdict    = "REAL" if output > 0.5 else "AI-GENERATED"
                    confidence = output if output > 0.5 else (1.0 - output)
                    st.markdown(f"**Discriminator Score:** `{output:.4f}`")
                    st.markdown(f"**Verdict:** `{verdict}`")
                    st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
                    st.progress(float(output))
                    real_pct = output * 100
                    ai_pct   = (1 - output) * 100
                    bar_html = f"""
                    <div style="margin-top:0.5rem;">
                      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                        <span style="width:100px;font-size:0.75rem;color:#a1a1aa;">REAL</span>
                        <div style="flex:1;background:#27272a;border-radius:4px;height:18px;overflow:hidden;">
                          <div style="width:{real_pct:.1f}%;background:linear-gradient(90deg,#22c55e,#4ade80);height:100%;border-radius:4px;"></div>
                        </div>
                        <span style="width:45px;font-size:0.75rem;color:#e4e4e7;">{real_pct:.1f}%</span>
                      </div>
                      <div style="display:flex;align-items:center;gap:8px;">
                        <span style="width:100px;font-size:0.75rem;color:#a1a1aa;">AI-GEN</span>
                        <div style="flex:1;background:#27272a;border-radius:4px;height:18px;overflow:hidden;">
                          <div style="width:{ai_pct:.1f}%;background:linear-gradient(90deg,#ef4444,#f87171);height:100%;border-radius:4px;"></div>
                        </div>
                        <span style="width:45px;font-size:0.75rem;color:#e4e4e7;">{ai_pct:.1f}%</span>
                      </div>
                    </div>"""
                    st.markdown(bar_html, unsafe_allow_html=True)
                    if verdict == "REAL":
                        st.success("Classified as REAL.")
                    else:
                        st.error("Classified as AI-GENERATED.")
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_scan(uploaded_file.name, output, verdict, ts)
        else:
            st.subheader(f"Batch Scan — {len(uploaded_files)} images")
            if st.button("Analyze All Images", use_container_width=True):
                results = []
                scores  = []
                with st.spinner("Analyzing batch..."):
                    for uf in uploaded_files:
                        img = Image.open(uf).convert("RGB").resize((64, 64))
                        img_tensor = img_transform(img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            score = disc(img_tensor).item()
                        verdict    = "REAL" if score > 0.5 else "AI-GENERATED"
                        confidence = score if score > 0.5 else (1.0 - score)
                        results.append({"filename": uf.name, "score": round(score, 4),
                                        "verdict": verdict, "confidence": f"{confidence*100:.1f}%"})
                        scores.append(score)
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_scan(uf.name, score, verdict, ts)

                st.markdown("**Results Table**")
                header_cols = st.columns([3, 2, 2, 2])
                header_cols[0].markdown("**Filename**")
                header_cols[1].markdown("**Score**")
                header_cols[2].markdown("**Verdict**")
                header_cols[3].markdown("**Confidence**")
                st.markdown("---")
                for r in results:
                    row = st.columns([3, 2, 2, 2])
                    row[0].write(r["filename"])
                    row[1].write(f'`{r["score"]}`')
                    vc = "#22c55e" if r["verdict"] == "REAL" else "#ef4444"
                    row[2].markdown(f'<span style="color:{vc};font-weight:600;">{r["verdict"]}</span>', unsafe_allow_html=True)
                    row[3].write(r["confidence"])

                st.markdown("---")
                st.subheader("Score Distribution")
                n_bins    = 10
                bin_edges = np.linspace(0, 1, n_bins + 1)
                counts, _ = np.histogram(scores, bins=bin_edges)
                max_count = max(counts) if max(counts) > 0 else 1
                hist_html = '<div style="display:flex;align-items:flex-end;gap:4px;height:120px;padding:8px;background:#18181b;border-radius:8px;border:1px solid #27272a;">'
                for i, cnt in enumerate(counts):
                    color = "#22c55e" if bin_edges[i] >= 0.5 else "#ef4444"
                    pct   = int((cnt / max_count) * 100)
                    hist_html += (f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:2px;">'
                                  f'<span style="font-size:0.6rem;color:#71717a;">{cnt}</span>'
                                  f'<div style="width:100%;height:{pct}%;background:{color};border-radius:3px 3px 0 0;min-height:2px;"></div>'
                                  f'<span style="font-size:0.55rem;color:#52525b;">{bin_edges[i]:.1f}</span></div>')
                hist_html += '</div>'
                st.markdown(hist_html, unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("Image Comparison")
                filenames = [r["filename"] for r in results]
                col_sel1, col_sel2 = st.columns(2)
                sel1 = col_sel1.selectbox("Image A", filenames, key="cmp_sel1")
                sel2 = col_sel2.selectbox("Image B", filenames, index=min(1, len(filenames)-1), key="cmp_sel2")
                col_c1, col_c2 = st.columns(2)
                for uf in uploaded_files:
                    if uf.name == sel1:
                        r_a = next(r for r in results if r["filename"] == sel1)
                        col_c1.image(Image.open(uf).convert("RGB").resize((180, 180), Image.NEAREST),
                                     caption=f"{sel1} | {r_a['verdict']} ({r_a['score']})", width=180)
                    if uf.name == sel2:
                        r_b = next(r for r in results if r["filename"] == sel2)
                        col_c2.image(Image.open(uf).convert("RGB").resize((180, 180), Image.NEAREST),
                                     caption=f"{sel2} | {r_b['verdict']} ({r_b['score']})", width=180)


# ══════════════════════════════════════════════════════════════
# PAGE: BATCH GENERATOR
# ══════════════════════════════════════════════════════════════
elif page == "Batch Generator":
    st.header("Batch Image Generation")
    st.write("Generate multiple images at once to explore the latent space.")

    if not model_loaded:
        st.error(f"Model error: {model_error}")
        st.stop()

    num_images = st.slider("Number of Images", min_value=1, max_value=16, value=4)
    base_seed  = st.number_input("Base Seed", value=0, min_value=0, step=1)

    if st.button("Generate Batch", use_container_width=True):
        cols = st.columns(4)
        batch_results = []
        with st.spinner("Generating batch..."):
            for i in range(num_images):
                current_seed = int(base_seed) + i
                torch.manual_seed(current_seed)
                noise = torch.randn(1, 100, 1, 1, device=device)
                with torch.no_grad():
                    img_tensor = gen(noise).detach().cpu()[0]
                img = postprocess_tensor(img_tensor).resize((128, 128), Image.NEAREST)
                batch_results.append((current_seed, img))
                cols[i % 4].image(img, caption=f"Seed {current_seed}", use_column_width=True)
                st.session_state.generated_images.append((current_seed, image_to_bytes(img)))
        st.session_state.batch_results = batch_results
        st.session_state.total_generated += num_images
        st.success(f"Generated {num_images} images.")
        st.markdown("---")
        st.subheader("Download Individual Results")
        dl_cols = st.columns(4)
        for idx, (s, img) in enumerate(batch_results):
            dl_cols[idx % 4].download_button(
                label=f"Seed {s}", data=image_to_bytes(img),
                file_name=f"batch_seed_{s}.png", mime="image/png")


# ══════════════════════════════════════════════════════════════
# PAGE: IMAGE GALLERY
# ══════════════════════════════════════════════════════════════
elif page == "Image Gallery":
    st.header("Image Gallery")
    st.write("All images generated during this session.")
    if not st.session_state.generated_images:
        st.info("No images generated yet.")
    else:
        total = len(st.session_state.generated_images)
        st.caption(f"{total} image{'s' if total != 1 else ''} in session")
        if st.button("🗑 Clear Gallery"):
            st.session_state.generated_images = []
            st.rerun()
        cols = st.columns(4)
        for i, (seed_label, img_bytes) in enumerate(reversed(st.session_state.generated_images)):
            col = cols[i % 4]
            img = Image.open(io.BytesIO(img_bytes))
            col.image(img, caption=f"Seed: {seed_label}", use_column_width=True)
            col.download_button("⬇", data=img_bytes, file_name=f"gallery_seed_{seed_label}.png",
                                mime="image/png", key=f"gal_dl_{i}", use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: GENERATION HISTORY
# ══════════════════════════════════════════════════════════════
elif page == "Generation History":
    st.header("Session History")
    tab_gen, tab_scan = st.tabs(["Generation Log", "Scan Log"])
    with tab_gen:
        st.subheader("Image Generation History")
        if st.session_state.generation_history:
            for entry in reversed(st.session_state.generation_history):
                st.markdown(f"- Seed `{entry['seed']}` at `{entry['timestamp']}`")
        else:
            st.info("No images generated this session.")
        if st.button("Clear Generation History"):
            st.session_state.generation_history = []
            st.success("Cleared.")
    with tab_scan:
        st.subheader("Image Scan History")
        if st.session_state.scan_history:
            for entry in reversed(st.session_state.scan_history):
                color = "#22c55e" if entry["verdict"] == "REAL" else "#ef4444"
                st.markdown(
                    f"- `{entry['file']}` — Score: `{entry['score']}` — "
                    f'<span style="color:{color};font-weight:600;">{entry["verdict"]}</span>'
                    f" — `{entry['timestamp']}`", unsafe_allow_html=True)
        else:
            st.info("No images scanned this session.")
        if st.button("Clear Scan History"):
            st.session_state.scan_history = []
            st.success("Cleared.")


# ══════════════════════════════════════════════════════════════
# ADVANCED: SEED EVOLUTION (GENETIC ALGORITHM)
# ══════════════════════════════════════════════════════════════
elif page == "Seed Evolution (GA)":
    st.header("Seed Evolution — Genetic Algorithm")
    st.write(
        "Rate each face as ❤ (keep) or ✖ (discard). "
        "The app breeds the next generation from seeds you liked — faces evolve toward your taste over time."
    )

    if not model_loaded:
        st.error(f"Model error: {model_error}")
        st.stop()

    POP_SIZE = 4

    def make_population(seeds):
        return [seed_to_latent(s, device) for s in seeds]

    def crossover(z_a, z_b):
        mask = torch.randint(0, 2, (100,), device=device).float()
        child = mask * z_a.squeeze() + (1 - mask) * z_b.squeeze()
        return child.unsqueeze(0)

    def mutate(z, rate=0.1):
        noise = torch.randn_like(z) * rate
        return z + noise

    def breed_next_generation(selected_zs, pop_size=POP_SIZE, mutate_rate=0.15):
        children = []
        n = len(selected_zs)
        if n == 0:
            seeds = [np.random.randint(0, 999999) for _ in range(pop_size)]
            return [seed_to_latent(s, device) for s in seeds]
        while len(children) < pop_size:
            idx_a, idx_b = np.random.choice(n, size=2, replace=(n < 2))
            child = crossover(selected_zs[idx_a], selected_zs[idx_b])
            child = mutate(child, mutate_rate)
            children.append(child)
        return children

    # Init population on first run
    if not st.session_state.evo_population:
        init_seeds = [np.random.randint(0, 999999) for _ in range(POP_SIZE)]
        st.session_state.evo_population = [seed_to_latent(s, device) for s in init_seeds]
        st.session_state.evo_generation = 1

    st.markdown(f"**Generation #{st.session_state.evo_generation}**")
    st.caption("Click ❤ to keep a face for breeding, ✖ to discard.")

    cols = st.columns(POP_SIZE)
    kept_flags = []

    for i, z in enumerate(st.session_state.evo_population):
        img_tensor = generate_image_from_z(z, gen, device)
        img = postprocess_tensor(img_tensor).resize((160, 160), Image.NEAREST)
        cols[i].image(img, use_column_width=True)
        keep = cols[i].checkbox("❤ Keep", key=f"evo_keep_{i}_{st.session_state.evo_generation}")
        kept_flags.append(keep)

    mutate_rate = st.slider("Mutation Rate", 0.01, 0.5, 0.15, 0.01,
                            help="Higher = more random variation in children")

    col_evo1, col_evo2 = st.columns(2)
    if col_evo1.button("➡ Next Generation", use_container_width=True):
        selected = [z for z, k in zip(st.session_state.evo_population, kept_flags) if k]
        st.session_state.evo_population = breed_next_generation(selected, POP_SIZE, mutate_rate)
        st.session_state.evo_generation += 1
        st.rerun()

    if col_evo2.button("🔄 Reset Evolution", use_container_width=True):
        st.session_state.evo_population = []
        st.session_state.evo_generation = 0
        st.rerun()

    st.markdown("---")
    st.subheader("Save a Face from This Generation")
    save_idx = st.selectbox("Select face to save", list(range(POP_SIZE)), format_func=lambda x: f"Face {x+1}")
    if st.button("💾 Save to Gallery", use_container_width=True):
        z = st.session_state.evo_population[save_idx]
        img_tensor = generate_image_from_z(z, gen, device)
        img = postprocess_tensor(img_tensor).resize((256, 256), Image.NEAREST)
        label = f"evo_gen{st.session_state.evo_generation}_face{save_idx+1}"
        st.session_state.generated_images.append((label, image_to_bytes(img)))
        log_generation(label, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.success(f"Saved {label} to gallery.")


# ══════════════════════════════════════════════════════════════
# ADVANCED: GAN INVERSION
# ══════════════════════════════════════════════════════════════
elif page == "GAN Inversion":
    st.header("GAN Inversion — Image to Latent Vector")
    st.write(
        "Upload an anime face image. The app runs an optimization loop to find "
        "the closest latent vector z that makes the Generator reproduce it. "
        "You can then fine-tune the result with the sliders below."
    )

    if not model_loaded:
        st.error(f"Model error: {model_error}")
        st.stop()

    uploaded = st.file_uploader("Upload Anime Face (PNG / JPG, ideally 64×64)", type=["png", "jpg", "jpeg"])

    col_p, col_r = st.columns(2)
    n_steps  = col_p.slider("Optimization Steps", 100, 1000, 300, 50)
    lr_inv   = col_r.select_slider("Learning Rate", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.01)

    if uploaded and st.button("▶ Run Inversion", use_container_width=True):
        target_img = Image.open(uploaded).convert("RGB").resize((64, 64))
        target_tensor = img_transform(target_img).unsqueeze(0).to(device)

        z = torch.randn(1, 100, 1, 1, device=device, requires_grad=True)
        optimizer = optim.Adam([z], lr=lr_inv)
        loss_fn   = nn.MSELoss()

        progress_bar = st.progress(0)
        loss_display = st.empty()

        for step in range(n_steps):
            optimizer.zero_grad()
            fake = gen(z)
            loss = loss_fn(fake, target_tensor)
            loss.backward()
            optimizer.step()
            if step % max(1, n_steps // 50) == 0:
                progress_bar.progress((step + 1) / n_steps)
                loss_display.caption(f"Step {step+1}/{n_steps} — Loss: {loss.item():.5f}")

        progress_bar.progress(1.0)
        loss_display.caption(f"Done! Final Loss: {loss.item():.5f}")

        z_final = z.detach().cpu()
        st.session_state.inverted_latent = z_final

        with torch.no_grad():
            reconstructed = gen(z_final.view(1, 100, 1, 1).to(device)).detach().cpu()[0]
        rec_img = postprocess_tensor(reconstructed).resize((256, 256), Image.NEAREST)
        st.session_state.inverted_image_bytes = image_to_bytes(rec_img)

        col1, col2 = st.columns(2)
        col1.image(target_img.resize((256, 256), Image.NEAREST), caption="Original", width=256)
        col2.image(rec_img, caption="Reconstructed", width=256)
        st.download_button("⬇ Download Reconstructed", data=image_to_bytes(rec_img),
                           file_name="inverted_reconstruction.png", mime="image/png")

    # Fine-tune sliders (only if inversion was run)
    if st.session_state.inverted_latent is not None:
        st.markdown("---")
        st.subheader("Fine-tune Inverted Latent with Sliders")
        st.caption("Adjust any of the first 10 dimensions of the found latent vector.")
        z_base = st.session_state.inverted_latent.squeeze()
        fine_dims = []
        slider_cols = st.columns(5)
        for d in range(10):
            col_idx = d % 5
            val = slider_cols[col_idx].slider(
                f"D{d}", -3.0, 3.0, float(z_base[d].item()), 0.05, key=f"inv_dim_{d}")
            fine_dims.append(val)

        if st.button("🔄 Regenerate with Adjustments", use_container_width=True):
            z_adj = z_base.clone()
            for d, v in enumerate(fine_dims):
                z_adj[d] = v
            with torch.no_grad():
                img_tensor = gen(z_adj.view(1, 100, 1, 1).to(device)).detach().cpu()[0]
            adj_img = postprocess_tensor(img_tensor).resize((256, 256), Image.NEAREST)
            st.image(adj_img, caption="Adjusted Reconstruction", width=256)
            st.download_button("⬇ Download Adjusted", data=image_to_bytes(adj_img),
                               file_name="inverted_adjusted.png", mime="image/png",
                               use_container_width=True)
            st.session_state.generated_images.append(("inv_adjusted", image_to_bytes(adj_img)))
            log_generation("inv_adjusted", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ══════════════════════════════════════════════════════════════
# ADVANCED: STYLE TRANSFER
# ══════════════════════════════════════════════════════════════
elif page == "Style Transfer":
    st.header("Style Transfer — Discriminator Feature Matching")
    st.write(
        "Upload a **content** image and a **style** image. "
        "The app optimizes a new image that matches the content's structure "
        "while borrowing the style's feature statistics (mean & std) from the Discriminator's layers."
    )

    if not model_loaded:
        st.error(f"Model error: {model_error}")
        st.stop()

    col_up1, col_up2 = st.columns(2)
    content_file = col_up1.file_uploader("Content Image", type=["png", "jpg", "jpeg"], key="st_content")
    style_file   = col_up2.file_uploader("Style Image",   type=["png", "jpg", "jpeg"], key="st_style")

    col_sp1, col_sp2, col_sp3 = st.columns(3)
    st_steps        = col_sp1.slider("Steps",          100, 800, 300, 50)
    content_weight  = col_sp2.slider("Content Weight", 0.1, 10.0, 1.0, 0.1)
    style_weight    = col_sp3.slider("Style Weight",   1.0, 100.0, 10.0, 1.0)

    def gram_matrix(feat):
        b, c, h, w = feat.size()
        f = feat.view(b, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)

    def get_disc_features(x, disc_model):
        """Extract features from discriminator layers (excluding final sigmoid layer)."""
        feats = []
        out = x
        for i, layer in enumerate(disc_model.main):
            out = layer(out)
            if i in [1, 4, 7, 10]:   # after each LeakyReLU block
                feats.append(out)
        return feats

    if content_file and style_file and st.button("▶ Run Style Transfer", use_container_width=True):
        content_img = Image.open(content_file).convert("RGB").resize((64, 64))
        style_img   = Image.open(style_file).convert("RGB").resize((64, 64))

        content_t = img_transform(content_img).unsqueeze(0).to(device)
        style_t   = img_transform(style_img).unsqueeze(0).to(device)

        # Target: start from content
        target = content_t.clone().requires_grad_(True)
        optimizer = optim.Adam([target], lr=0.02)

        with torch.no_grad():
            content_feats = get_disc_features(content_t, disc)
            style_feats   = get_disc_features(style_t, disc)
            style_grams   = [gram_matrix(f) for f in style_feats]

        progress = st.progress(0)
        status   = st.empty()

        for step in range(st_steps):
            optimizer.zero_grad()
            target_feats = get_disc_features(target, disc)

            c_loss = sum(nn.MSELoss()(tf, cf.detach())
                         for tf, cf in zip(target_feats, content_feats)) * content_weight

            s_loss = sum(nn.MSELoss()(gram_matrix(tf), sg.detach())
                         for tf, sg in zip(target_feats, style_grams)) * style_weight

            loss = c_loss + s_loss
            loss.backward()
            optimizer.step()
            # Clamp to valid range
            with torch.no_grad():
                target.clamp_(-1, 1)

            if step % max(1, st_steps // 40) == 0:
                progress.progress((step + 1) / st_steps)
                status.caption(f"Step {step+1}/{st_steps} — Loss: {loss.item():.4f}")

        progress.progress(1.0)
        result_tensor = target.detach().cpu()[0]
        result_img    = postprocess_tensor(result_tensor).resize((256, 256), Image.NEAREST)

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.image(content_img.resize((200, 200)), caption="Content", width=200)
        col_r2.image(style_img.resize((200, 200)),   caption="Style",   width=200)
        col_r3.image(result_img,                     caption="Result",  width=200)

        st.download_button("⬇ Download Result", data=image_to_bytes(result_img),
                           file_name="style_transfer_result.png", mime="image/png",
                           use_container_width=True)
        st.session_state.generated_images.append(("style_transfer", image_to_bytes(result_img)))
        log_generation("style_transfer", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        status.success("Style Transfer complete!")


# ══════════════════════════════════════════════════════════════
# ADVANCED: GIF EXPORT
# ══════════════════════════════════════════════════════════════
elif page == "GIF Export":
    st.header("Animation / GIF Export")
    st.write(
        "Choose a start and end seed. The app generates a smooth SLERP interpolation "
        "and exports it as a downloadable animated GIF."
    )

    if not model_loaded:
        st.error(f"Model error: {model_error}")
        st.stop()

    col_a, col_b = st.columns(2)
    gif_seed_a = col_a.number_input("Start Seed", value=42,  min_value=0, max_value=999999, step=1)
    gif_seed_b = col_b.number_input("End Seed",   value=137, min_value=0, max_value=999999, step=1)

    col_c, col_d, col_e = st.columns(3)
    gif_frames = col_c.slider("Frames",    8, 60, 24)
    gif_fps    = col_d.slider("FPS",       4, 30, 12)
    gif_size   = col_e.select_slider("Frame Size (px)", options=[64, 96, 128, 192, 256], value=128)

    if st.button("🎬 Generate & Export GIF", use_container_width=True):
        z_a = seed_to_latent(gif_seed_a, device)
        z_b = seed_to_latent(gif_seed_b, device)

        frames = []
        progress = st.progress(0)

        with st.spinner("Rendering frames..."):
            for i in range(gif_frames):
                alpha = i / max(gif_frames - 1, 1)
                z = slerp(alpha, z_a, z_b).view(1, 100, 1, 1)
                with torch.no_grad():
                    img_tensor = gen(z).detach().cpu()[0]
                frame = postprocess_tensor(img_tensor).resize((gif_size, gif_size), Image.NEAREST)
                frames.append(np.array(frame))
                progress.progress((i + 1) / gif_frames)

        # Build GIF in memory
        gif_buf = io.BytesIO()
        duration = 1000 // gif_fps   # ms per frame
        imageio.mimsave(gif_buf, frames, format="GIF", duration=duration, loop=0)
        gif_bytes = gif_buf.getvalue()

        st.success(f"Generated {gif_frames}-frame GIF at {gif_fps} FPS ({gif_size}×{gif_size}px)")

        # Preview first/last frame
        col_p1, col_p2 = st.columns(2)
        col_p1.image(frames[0],  caption="First Frame", width=gif_size)
        col_p2.image(frames[-1], caption="Last Frame",  width=gif_size)

        st.download_button(
            "⬇ Download Animated GIF",
            data=gif_bytes,
            file_name=f"anime_interpolation_{gif_seed_a}_to_{gif_seed_b}.gif",
            mime="image/gif",
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════
# ADVANCED: DISCRIMINATOR FEATURE MAP VISUALIZER
# ══════════════════════════════════════════════════════════════
elif page == "Feature Map Visualizer":
    st.header("Discriminator Feature Map Visualizer")
    st.write(
        "Upload an anime image and see the activation heatmaps from each convolutional block "
        "of the Discriminator — understand what the model 'sees' when it classifies an image."
    )

    if not model_loaded:
        st.error(f"Model error: {model_error}")
        st.stop()

    fv_file = st.file_uploader("Upload Image (PNG / JPG)", type=["png", "jpg", "jpeg"])
    max_channels = st.slider("Channels to Show per Layer", 4, 16, 8,
                             help="Show the first N activation maps from each block")

    def featuremap_to_heatmap(fmap_2d):
        """Normalize a single 2D feature map to a displayable RGB heatmap."""
        arr = fmap_2d.numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).astype(np.uint8)
        # Apply colormap manually (viridis-like: blue → green → yellow)
        r = (arr * 2).clip(0, 255).astype(np.uint8)
        g = arr
        b = (255 - arr).clip(0, 255).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
        return rgb

    if fv_file and st.button("▶ Visualize Feature Maps", use_container_width=True):
        img = Image.open(fv_file).convert("RGB").resize((64, 64))
        img_tensor = img_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            score = disc(img_tensor).item()
        verdict = "REAL" if score > 0.5 else "AI-GENERATED"

        st.markdown(f"**Score:** `{score:.4f}` — **Verdict:** `{verdict}`")
        st.image(img.resize((128, 128), Image.NEAREST), caption="Input Image", width=128)
        st.markdown("---")

        features, layer_names = disc.get_feature_maps(img_tensor)

        for feat, name in zip(features, layer_names):
            st.subheader(name)
            n_ch     = min(max_channels, feat.shape[1])
            feat_map = feat[0]   # shape: (C, H, W)

            # Show channels as heatmaps
            grid_cols = st.columns(n_ch)
            for ch_idx in range(n_ch):
                fmap_2d = feat_map[ch_idx]     # (H, W)
                heatmap  = featuremap_to_heatmap(fmap_2d)
                # Upscale for visibility
                heatmap_img = Image.fromarray(heatmap).resize((80, 80), Image.NEAREST)
                grid_cols[ch_idx].image(heatmap_img, caption=f"Ch {ch_idx}", use_column_width=True)

            # Also show mean activation map
            mean_map = feat_map.mean(0)     # average over all channels
            mean_heat = featuremap_to_heatmap(mean_map)
            mean_img  = Image.fromarray(mean_heat).resize((160, 160), Image.NEAREST)
            st.image(mean_img, caption=f"Mean Activation — {name}", width=160)
            st.caption(f"Shape: {tuple(feat.shape)} | Mean: {feat.mean().item():.4f} | Std: {feat.std().item():.4f}")
            st.markdown("---")
