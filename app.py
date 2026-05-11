import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import time
import json
import csv
import os
from datetime import datetime

# --- Page Config ---
st.set_page_config(
    page_title="AnimeGen Pro AI",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme Management ---
def get_theme():
    return st.session_state.get("theme", "dark")

def load_css():
    theme = get_theme()
    with open("style.css") as f:
        css = f.read()
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

# --- Model Architectures ---
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


def _remap_discriminator_state_dict(state_dict):
    """
    Remaps old checkpoint keys (e.g. 'main.0.weight') to match the
    current Discriminator architecture which also uses 'main.*'.
    Handles cases where the checkpoint was saved with a different
    wrapper (e.g. 'layers.*' / 'final.*') by converting them back.
    """
    # If keys already match 'main.*' format, return as-is
    if any(k.startswith("main.") for k in state_dict.keys()):
        return state_dict

    remapped = {}
    # Map old 'layers.X.Y.*' / 'final.X.*' → 'main.N.*'
    # Architecture: main indices [0,2,3,5,6,8,9,11] for weights/BN,
    # interleaved with LeakyReLU (no params).
    # layers.0.0  → main.0   (Conv)
    # layers.1.0  → main.2   (Conv)  layers.1.1.* → main.3.*  (BN)
    # layers.2.0  → main.5   (Conv)  layers.2.1.* → main.6.*  (BN)
    # layers.3.0  → main.8   (Conv)  layers.3.1.* → main.9.*  (BN)
    # final.0.*   → main.11.*        (Conv)
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

    disc_state = checkpoint["modelD_state_dict"]
    disc_state = _remap_discriminator_state_dict(disc_state)
    disc.load_state_dict(disc_state)

    gen.to(device).eval()
    disc.to(device).eval()
    return gen, disc, device


# --- Session State Initialization ---
def init_session_state():
    defaults = {
        "generated_images": [],       # list of (seed, bytes)
        "generation_history": [],
        "scan_history": [],
        "favorite_seeds": [],
        "batch_results": [],
        "total_generated": 0,
        "total_scanned": 0,
        "theme": "dark",
        # Feature 3: latent sliders state
        "latent_dims": [0.0] * 10,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()
load_css()

# --- Utilities ---
def postprocess_tensor(img_tensor):
    arr = ((img_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)

def apply_image_adjustments(img, brightness=1.0, contrast=1.0, sharpness=1.0):
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img

def image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def log_generation(seed, timestamp):
    entry = {"seed": seed, "timestamp": timestamp}
    st.session_state.generation_history.append(entry)
    st.session_state.total_generated += 1

def log_scan(filename, score, verdict, timestamp):
    entry = {"file": filename, "score": round(score, 4), "verdict": verdict, "timestamp": timestamp}
    st.session_state.scan_history.append(entry)
    st.session_state.total_scanned += 1

def slerp(val, low, high):
    """Spherical linear interpolation between two latent vectors."""
    low_norm = low / (torch.norm(low, dim=1, keepdim=True) + 1e-8)
    high_norm = high / (torch.norm(high, dim=1, keepdim=True) + 1e-8)
    omega = torch.acos((low_norm * high_norm).sum(1).clamp(-1, 1))
    so = torch.sin(omega)
    if so.item() < 1e-8:
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + \
           (torch.sin(val * omega) / so).unsqueeze(1) * high

# --- Load Models ---
try:
    gen, disc, device = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# --- Sidebar ---
st.sidebar.title("AnimeGen Pro AI")
st.sidebar.caption("GAN-based Anime Face Generation & Authentication")

# Feature 8: Dark/Light Mode Toggle
theme_label = "☀ Light Mode" if get_theme() == "dark" else "🌑 Dark Mode"
if st.sidebar.button(theme_label, use_container_width=True):
    st.session_state.theme = "light" if get_theme() == "dark" else "dark"
    st.rerun()

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["AI Generator", "Vision Scanner", "Batch Generator", "Image Gallery", "Generation History"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Session Stats**")
st.sidebar.metric("Images Generated", st.session_state.total_generated)
st.sidebar.metric("Images Scanned", st.session_state.total_scanned)
st.sidebar.metric("Favorites Saved", len(st.session_state.favorite_seeds))
st.sidebar.markdown(f"**Device:** `{'CUDA' if torch.cuda.is_available() else 'CPU'}`")
st.sidebar.markdown(f"**Model:** `{'Loaded ✓' if model_loaded else 'Not Found ✗'}`")

# Feature 9: Export Session Report
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
    else:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["section", "seed_or_file", "score", "verdict", "timestamp"])
        for e in st.session_state.generation_history:
            writer.writerow(["generation", e["seed"], "", "", e["timestamp"]])
        for e in st.session_state.scan_history:
            writer.writerow(["scan", e["file"], e["score"], e["verdict"], e["timestamp"]])
        return buf.getvalue().encode(), "session_report.csv", "text/csv"

export_data, export_name, export_mime = build_export()
st.sidebar.download_button(
    "⬇ Export Report",
    data=export_data,
    file_name=export_name,
    mime=export_mime,
    use_container_width=True
)


# ==================== AI GENERATOR ====================
if page == "AI Generator":
    st.header("Image Synthesis")

    if not model_loaded:
        st.error(f"Model could not be loaded: {model_error}")
        st.stop()

    gen_mode = st.radio(
        "Generation Mode",
        ["Standard", "Interpolation", "Noise Grid", "Latent Sliders"],
        horizontal=True
    )

    st.markdown("---")

    # ── Standard ──────────────────────────────────────────────
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

    # ── Feature 1: Interpolation ───────────────────────────────
    elif gen_mode == "Interpolation":
        st.subheader("Seed Interpolation")
        st.write("Generate a smooth transition between two different faces.")

        col_a, col_b, col_c = st.columns(3)
        seed_a = col_a.number_input("Seed A", value=42,   min_value=0, max_value=999999, step=1)
        seed_b = col_b.number_input("Seed B", value=137,  min_value=0, max_value=999999, step=1)
        steps  = col_c.slider("Interpolation Steps", min_value=3, max_value=12, value=7)

        if st.button("Generate Interpolation", use_container_width=True):
            torch.manual_seed(int(seed_a))
            z_a = torch.randn(1, 100, device=device)
            torch.manual_seed(int(seed_b))
            z_b = torch.randn(1, 100, device=device)

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

    # ── Feature 2: Noise Exploration Grid ─────────────────────
    elif gen_mode == "Noise Grid":
        st.subheader("Noise Exploration Grid")
        st.write("9 variations around a base seed — slight noise perturbations reveal latent diversity.")

        seed_base  = st.number_input("Base Seed", value=42, min_value=0, max_value=999999, step=1)
        noise_scale = st.slider("Perturbation Scale", 0.01, 0.5, 0.1, 0.01)

        if st.button("Generate Grid", use_container_width=True):
            torch.manual_seed(int(seed_base))
            z_base = torch.randn(1, 100, 1, 1, device=device)

            cols = st.columns(3)
            with st.spinner("Generating 9-image grid..."):
                for i in range(9):
                    perturb = torch.randn_like(z_base) * noise_scale
                    z = z_base + perturb
                    with torch.no_grad():
                        img_tensor = gen(z).detach().cpu()[0]
                    img = postprocess_tensor(img_tensor).resize((128, 128), Image.NEAREST)
                    label = "Base" if i == 0 else f"Var {i}"
                    if i == 0:
                        z = z_base  # center cell is pure base
                        with torch.no_grad():
                            img_tensor = gen(z_base).detach().cpu()[0]
                        img = postprocess_tensor(img_tensor).resize((128, 128), Image.NEAREST)
                    cols[i % 3].image(img, caption=label, use_column_width=True)

            st.success("Noise grid complete.")

    # ── Feature 3: Latent Vector Sliders ──────────────────────
    elif gen_mode == "Latent Sliders":
        st.subheader("Latent Vector Sliders")
        st.write("Manually control 10 dimensions of the 100-dim noise vector. Remaining 90 dims are fixed by seed.")

        col_sliders, col_preview = st.columns([1, 1])

        with col_sliders:
            base_seed = st.number_input("Base Seed (fixes 90 dims)", value=42, min_value=0, max_value=999999, step=1)
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


# ==================== VISION SCANNER ====================
elif page == "Vision Scanner":
    st.header("Vision Scanner — AI Image Authentication")

    if not model_loaded:
        st.error(f"Model could not be loaded: {model_error}")
        st.stop()

    st.write(
        "Upload one or multiple anime face images. The discriminator scores each image: "
        "values above 0.5 → Real, below 0.5 → AI-Generated."
    )

    # Feature 4: Batch Image Scan
    uploaded_files = st.file_uploader(
        "Upload Images (JPG / PNG, 64×64 preferred) — multi-select supported",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if uploaded_files:
        if len(uploaded_files) == 1:
            # ── Single file: original detailed view ──────────────
            uploaded_file = uploaded_files[0]
            img = Image.open(uploaded_file).convert("RGB").resize((64, 64))
            col_img, col_res = st.columns([1, 1])

            with col_img:
                st.image(img.resize((200, 200), Image.NEAREST), caption="Input Image", width=200)
                st.download_button(
                    label="⬇ Download Resized",
                    data=image_to_bytes(img),
                    file_name=f"scanned_{uploaded_file.name}",
                    mime="image/png"
                )

            with col_res:
                if st.button("Analyze Image", use_container_width=True):
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = disc(img_tensor).item()

                    verdict = "REAL" if output > 0.5 else "AI-GENERATED"
                    confidence = output if output > 0.5 else (1.0 - output)

                    st.markdown(f"**Discriminator Score:** `{output:.4f}`")
                    st.markdown(f"**Verdict:** `{verdict}`")
                    st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
                    st.progress(float(output))
                    st.caption("Score near 1.0 = Real   |   Score near 0.0 = AI-Generated")

                    # Feature 5: Confidence Breakdown Bar Chart
                    st.markdown("**Confidence Breakdown**")
                    real_pct     = output * 100
                    ai_pct       = (1 - output) * 100
                    bar_html = f"""
                    <div style="margin-top:0.5rem;">
                      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                        <span style="width:100px;font-size:0.75rem;font-family:monospace;color:#a1a1aa;">REAL</span>
                        <div style="flex:1;background:#27272a;border-radius:4px;height:18px;overflow:hidden;">
                          <div style="width:{real_pct:.1f}%;background:linear-gradient(90deg,#22c55e,#4ade80);height:100%;border-radius:4px;transition:width 0.4s;"></div>
                        </div>
                        <span style="width:45px;font-size:0.75rem;font-family:monospace;color:#e4e4e7;">{real_pct:.1f}%</span>
                      </div>
                      <div style="display:flex;align-items:center;gap:8px;">
                        <span style="width:100px;font-size:0.75rem;font-family:monospace;color:#a1a1aa;">AI-GEN</span>
                        <div style="flex:1;background:#27272a;border-radius:4px;height:18px;overflow:hidden;">
                          <div style="width:{ai_pct:.1f}%;background:linear-gradient(90deg,#ef4444,#f87171);height:100%;border-radius:4px;transition:width 0.4s;"></div>
                        </div>
                        <span style="width:45px;font-size:0.75rem;font-family:monospace;color:#e4e4e7;">{ai_pct:.1f}%</span>
                      </div>
                    </div>
                    """
                    st.markdown(bar_html, unsafe_allow_html=True)

                    if verdict == "REAL":
                        st.success("Classified as REAL.")
                    else:
                        st.error("Classified as AI-GENERATED.")

                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_scan(uploaded_file.name, output, verdict, ts)

        else:
            # ── Batch scan ────────────────────────────────────────
            st.subheader(f"Batch Scan — {len(uploaded_files)} images")
            if st.button("Analyze All Images", use_container_width=True):
                results = []
                scores  = []

                with st.spinner("Analyzing batch..."):
                    for uf in uploaded_files:
                        img = Image.open(uf).convert("RGB").resize((64, 64))
                        img_tensor = transform(img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            score = disc(img_tensor).item()
                        verdict    = "REAL" if score > 0.5 else "AI-GENERATED"
                        confidence = score if score > 0.5 else (1.0 - score)
                        results.append({
                            "filename": uf.name,
                            "score":    round(score, 4),
                            "verdict":  verdict,
                            "confidence": f"{confidence * 100:.1f}%"
                        })
                        scores.append(score)
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_scan(uf.name, score, verdict, ts)

                # Table
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
                    verdict_color = "#22c55e" if r["verdict"] == "REAL" else "#ef4444"
                    row[2].markdown(f'<span style="color:{verdict_color};font-weight:600;">{r["verdict"]}</span>', unsafe_allow_html=True)
                    row[3].write(r["confidence"])

                # Feature 6: Score Distribution Histogram
                st.markdown("---")
                st.subheader("Score Distribution")
                n_bins   = 10
                bin_edges = np.linspace(0, 1, n_bins + 1)
                counts, _ = np.histogram(scores, bins=bin_edges)
                max_count = max(counts) if max(counts) > 0 else 1
                bar_width = 100 // n_bins

                hist_html = '<div style="display:flex;align-items:flex-end;gap:4px;height:120px;padding:8px;background:#18181b;border-radius:8px;border:1px solid #27272a;">'
                for i, cnt in enumerate(counts):
                    left_edge = bin_edges[i]
                    color = "#22c55e" if left_edge >= 0.5 else "#ef4444"
                    pct   = int((cnt / max_count) * 100)
                    hist_html += f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:2px;">'
                    hist_html += f'  <span style="font-size:0.6rem;color:#71717a;">{cnt}</span>'
                    hist_html += f'  <div style="width:100%;height:{pct}%;background:{color};border-radius:3px 3px 0 0;min-height:2px;"></div>'
                    hist_html += f'  <span style="font-size:0.55rem;color:#52525b;">{left_edge:.1f}</span>'
                    hist_html += f'</div>'
                hist_html += '</div><div style="margin-top:4px;font-size:0.7rem;color:#71717a;font-family:monospace;text-align:center;">Score Distribution (green = Real zone, red = AI zone)</div>'
                st.markdown(hist_html, unsafe_allow_html=True)

                # Feature 7: Side-by-Side Image Comparison
                st.markdown("---")
                st.subheader("Image Comparison")
                filenames = [r["filename"] for r in results]
                col_sel1, col_sel2 = st.columns(2)
                sel1 = col_sel1.selectbox("Image A", filenames, key="cmp_sel1")
                sel2 = col_sel2.selectbox("Image B", filenames, index=min(1, len(filenames)-1), key="cmp_sel2")

                col_c1, col_c2 = st.columns(2)
                for uf in uploaded_files:
                    if uf.name == sel1:
                        img_a = Image.open(uf).convert("RGB").resize((64, 64))
                        r_a   = next(r for r in results if r["filename"] == sel1)
                        col_c1.image(img_a.resize((180, 180), Image.NEAREST), caption=f"{sel1} | {r_a['verdict']} ({r_a['score']})", width=180)
                    if uf.name == sel2:
                        img_b = Image.open(uf).convert("RGB").resize((64, 64))
                        r_b   = next(r for r in results if r["filename"] == sel2)
                        col_c2.image(img_b.resize((180, 180), Image.NEAREST), caption=f"{sel2} | {r_b['verdict']} ({r_b['score']})", width=180)


# ==================== BATCH GENERATOR ====================
elif page == "Batch Generator":
    st.header("Batch Image Generation")
    st.write("Generate multiple images at once to explore the latent space.")

    if not model_loaded:
        st.error(f"Model could not be loaded: {model_error}")
        st.stop()

    num_images = st.slider("Number of Images", min_value=1, max_value=16, value=4)
    base_seed  = st.number_input("Base Seed (consecutive seeds used)", value=0, min_value=0, step=1)

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
                label=f"Seed {s}",
                data=image_to_bytes(img),
                file_name=f"batch_seed_{s}.png",
                mime="image/png"
            )


# ==================== Feature 10: IMAGE GALLERY ====================
elif page == "Image Gallery":
    st.header("Image Gallery")
    st.write("All images generated during this session.")

    if not st.session_state.generated_images:
        st.info("No images generated yet. Head to the AI Generator or Batch Generator first.")
    else:
        total = len(st.session_state.generated_images)
        st.caption(f"{total} image{'s' if total != 1 else ''} in session")

        if st.button("🗑 Clear Gallery", use_container_width=False):
            st.session_state.generated_images = []
            st.rerun()

        cols = st.columns(4)
        for i, (seed_label, img_bytes) in enumerate(reversed(st.session_state.generated_images)):
            col = cols[i % 4]
            img = Image.open(io.BytesIO(img_bytes))
            col.image(img, caption=f"Seed: {seed_label}", use_column_width=True)
            col.download_button(
                label="⬇",
                data=img_bytes,
                file_name=f"gallery_seed_{seed_label}.png",
                mime="image/png",
                key=f"gal_dl_{i}",
                use_container_width=True
            )


# ==================== GENERATION HISTORY ====================
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
                    f" — `{entry['timestamp']}`",
                    unsafe_allow_html=True
                )
        else:
            st.info("No images scanned this session.")
        if st.button("Clear Scan History"):
            st.session_state.scan_history = []
            st.success("Cleared.")
