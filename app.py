import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import time
import json
import os
from datetime import datetime

# --- Page Config ---
st.set_page_config(
    page_title="AnimeGen Pro AI",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Custom CSS ---
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = Generator()
    disc = Discriminator()
    checkpoint = torch.load("best_gan_model.pth", map_location=device)
    gen.load_state_dict(checkpoint["modelG_state_dict"])
    disc.load_state_dict(checkpoint["modelD_state_dict"])
    gen.to(device).eval()
    disc.to(device).eval()
    return gen, disc, device


# --- Feature 1: Session State Initialization ---
def init_session_state():
    defaults = {
        "generated_images": [],
        "generation_history": [],
        "scan_history": [],
        "favorite_seeds": [],
        "batch_results": [],
        "total_generated": 0,
        "total_scanned": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# --- Feature 2: Image Post-Processing Utilities ---
def postprocess_tensor(img_tensor):
    """Convert generator output tensor to PIL Image."""
    arr = ((img_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def apply_image_adjustments(img: Image.Image, brightness=1.0, contrast=1.0, sharpness=1.0) -> Image.Image:
    """Apply post-processing adjustments to a PIL Image."""
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img


# --- Feature 3: Image Download Utility ---
def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- Feature 4: History Logger ---
def log_generation(seed: int, timestamp: str):
    entry = {"seed": seed, "timestamp": timestamp}
    st.session_state.generation_history.append(entry)
    st.session_state.total_generated += 1


def log_scan(filename: str, score: float, verdict: str, timestamp: str):
    entry = {"file": filename, "score": round(score, 4), "verdict": verdict, "timestamp": timestamp}
    st.session_state.scan_history.append(entry)
    st.session_state.total_scanned += 1


# --- Load Models ---
try:
    gen, disc, device = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# --- Sidebar ---
st.sidebar.title("AnimeGen Pro AI")
st.sidebar.caption("GAN-based Anime Face Generation and Authentication")

page = st.sidebar.radio(
    "Navigation",
    ["AI Generator", "Vision Scanner", "Batch Generator", "Generation History"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Session Stats**")
st.sidebar.metric("Images Generated", st.session_state.total_generated)
st.sidebar.metric("Images Scanned", st.session_state.total_scanned)
st.sidebar.metric("Favorites Saved", len(st.session_state.favorite_seeds))
st.sidebar.markdown(f"**Device:** `{'CUDA' if torch.cuda.is_available() else 'CPU'}`")
st.sidebar.markdown(f"**Model Status:** `{'Loaded' if model_loaded else 'Not Found'}`")

# ==================== AI GENERATOR ====================
if page == "AI Generator":
    st.header("Image Synthesis")

    if not model_loaded:
        st.error(f"Model could not be loaded: {model_error}")
        st.stop()

    col_ctrl, col_out = st.columns([1, 1])

    with col_ctrl:
        st.subheader("Controls")
        seed = st.number_input("Random Seed", value=42, min_value=0, max_value=999999, step=1)
        use_random_seed = st.checkbox("Use random seed on each run", value=False)

        st.markdown("**Post-Processing Adjustments**")
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05)
        sharpness = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.1)

        generate_btn = st.button("Generate Image", use_container_width=True)

        # Feature 7: Favorite Seeds
        if st.session_state.favorite_seeds:
            st.markdown("**Saved Favorite Seeds**")
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
            img_upscaled = img.resize((256, 256), Image.NEAREST)

            st.image(img_upscaled, caption=f"Seed: {actual_seed}", use_column_width=False, width=256)

            # Feature 5: Download
            st.download_button(
                label="Download PNG",
                data=image_to_bytes(img_upscaled),
                file_name=f"anime_seed_{actual_seed}.png",
                mime="image/png",
                use_container_width=True
            )

            # Feature 7: Save favorite
            if st.button("Save Seed to Favorites"):
                if actual_seed not in st.session_state.favorite_seeds:
                    st.session_state.favorite_seeds.append(actual_seed)
                    st.success(f"Seed {actual_seed} saved to favorites.")

            # Feature 4: Log
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_generation(actual_seed, ts)
            st.session_state.generated_images.append(image_to_bytes(img_upscaled))

# ==================== VISION SCANNER ====================
elif page == "Vision Scanner":
    st.header("Vision Scanner — AI Image Authentication")

    if not model_loaded:
        st.error(f"Model could not be loaded: {model_error}")
        st.stop()

    st.write(
        "Upload an anime face image to determine whether it is a real photograph "
        "or an AI-generated image. The discriminator outputs a score between 0 and 1: "
        "values above 0.5 indicate a real image; values below 0.5 indicate AI generation."
    )

    uploaded_file = st.file_uploader("Upload Image (JPG or PNG, 64x64 preferred)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB").resize((64, 64))
        col_img, col_res = st.columns([1, 1])

        with col_img:
            st.image(img.resize((200, 200), Image.NEAREST), caption="Input Image", width=200)
            # Feature 5: Download scanned image
            st.download_button(
                label="Download Resized Image",
                data=image_to_bytes(img),
                file_name=f"scanned_{uploaded_file.name}",
                mime="image/png"
            )

        with col_res:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            img_tensor = transform(img).unsqueeze(0).to(device)

            if st.button("Analyze Image", use_container_width=True):
                with torch.no_grad():
                    output = disc(img_tensor).item()

                verdict = "REAL" if output > 0.5 else "AI-GENERATED"
                confidence = output if output > 0.5 else (1.0 - output)

                st.markdown(f"**Discriminator Score:** `{output:.4f}`")
                st.markdown(f"**Verdict:** `{verdict}`")
                st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")

                # Feature 8: Confidence meter
                st.progress(float(output))
                st.caption("Score near 1.0 = Real   |   Score near 0.0 = AI-Generated")

                if verdict == "REAL":
                    st.success("This image is classified as REAL.")
                else:
                    st.error("This image is classified as AI-GENERATED.")

                # Feature 4: Log scan
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_scan(uploaded_file.name, output, verdict, ts)

# ==================== BATCH GENERATOR (Feature 3) ====================
elif page == "Batch Generator":
    st.header("Batch Image Generation")
    st.write("Generate multiple images at once to explore the latent space.")

    if not model_loaded:
        st.error(f"Model could not be loaded: {model_error}")
        st.stop()

    num_images = st.slider("Number of Images to Generate", min_value=1, max_value=16, value=4)
    base_seed = st.number_input("Base Seed (images use consecutive seeds)", value=0, min_value=0, step=1)

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
                col = cols[i % 4]
                col.image(img, caption=f"Seed {current_seed}", use_column_width=True)

        st.session_state.batch_results = batch_results
        st.session_state.total_generated += num_images
        st.success(f"Generated {num_images} images.")

        # Download all as zip concept — offer individual downloads
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

# ==================== GENERATION HISTORY (Feature 5 / 4) ====================
elif page == "Generation History":
    st.header("Session History")

    tab_gen, tab_scan = st.tabs(["Generation Log", "Scan Log"])

    with tab_gen:
        st.subheader("Image Generation History")
        if st.session_state.generation_history:
            for entry in reversed(st.session_state.generation_history):
                st.markdown(f"- Seed `{entry['seed']}` at `{entry['timestamp']}`")
        else:
            st.info("No images have been generated in this session yet.")

        if st.button("Clear Generation History"):
            st.session_state.generation_history = []
            st.success("Generation history cleared.")

    with tab_scan:
        st.subheader("Image Scan History")
        if st.session_state.scan_history:
            for entry in reversed(st.session_state.scan_history):
                verdict_tag = "REAL" if entry["verdict"] == "REAL" else "AI-GENERATED"
                st.markdown(
                    f"- `{entry['file']}` — Score: `{entry['score']}` — "
                    f"Verdict: **{verdict_tag}** — `{entry['timestamp']}`"
                )
        else:
            st.info("No images have been scanned in this session yet.")

        if st.button("Clear Scan History"):
            st.session_state.scan_history = []
            st.success("Scan history cleared.")
