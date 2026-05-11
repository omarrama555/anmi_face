import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image
import io
import os
import time
import zipfile
import random

# --- Architectures ---
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input): return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid()
        )
    def forward(self, input): return self.main(input)

# --- App Config ---
st.set_page_config(page_title="AnimeGen Pro", layout="wide")
if not os.path.exists("outputs"): os.makedirs("outputs")

# Load CSS
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen, disc = Generator(), Discriminator()
    path = "best_gan_model.pth"
    
    if not os.path.exists(path):
        return None, None, device, False

    try:
        ckpt = torch.load(path, map_location=device)
        gen.load_state_dict(ckpt["modelG_state_dict"])
        disc.load_state_dict(ckpt["modelD_state_dict"])
        return gen.to(device).eval(), disc.to(device).eval(), device, True
    except Exception as e:
        st.error(f"Error loading weights: {e}")
        return None, None, device, False

# --- Auth & State ---
if 'auth' not in st.session_state: st.session_state.auth = False
if 'history' not in st.session_state: st.session_state.history = []

if not st.session_state.auth:
    st.title("System Login")
    u, p = st.text_input("Username"), st.text_input("Password", type="password")
    if st.button("Login"):
        if u == "admin" and p == "admin":
            st.session_state.auth = True
            st.rerun()
        else: st.error("Access Denied")
else:
    gen, disc, device, loaded = load_assets()
    
    if not loaded:
        st.error("Missing 'best_gan_model.pth'. Ensure the file is in the same folder as app.py")
        st.stop()

    page = st.sidebar.radio("Navigation", ["Generator", "Batch Gen", "Vision Scanner", "Gallery"])
    if st.sidebar.button("Logout"):
        st.session_state.auth = False
        st.rerun()

    if page == "Generator":
        st.header("Single Image Generation")
        col1, col2 = st.columns([1, 2])
        with col1:
            seed = st.number_input("Seed", value=random.randint(0, 9999))
            noise_scale = st.slider("Noise Scale", 0.1, 2.0, 1.0)
            if st.button("Run Generator"):
                with st.spinner("Processing..."):
                    torch.manual_seed(seed)
                    noise = torch.randn(1, 100, 1, 1, device=device) * noise_scale
                    with torch.no_grad():
                        out = gen(noise).detach().cpu()[0]
                    img = Image.fromarray(((out.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).astype('uint8'))
                    st.session_state.history.append(img)
                    st.toast("Generation Complete")
        
        with col2:
            # SAFETY CHECK: Only display if history is NOT empty
            if st.session_state.history:
                latest_img = st.session_state.history[-1]
                st.image(latest_img, width=400)
                buf = io.BytesIO()
                latest_img.save(buf, format="PNG")
                st.download_button("Download Image", buf.getvalue(), "anime.png")
            else:
                st.info("Generation ready. Click 'Run Generator' to start.")

    elif page == "Batch Gen":
        st.header("Batch Export")
        num = st.select_slider("Count", options=[4, 8, 16])
        if st.button("Generate Batch"):
            noise = torch.randn(num, 100, 1, 1, device=device)
            with torch.no_grad():
                fakes = gen(noise).detach().cpu()
            grid = utils.make_grid(fakes, nrow=4, normalize=True)
            st.image(Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype('uint8')))

    elif page == "Vision Scanner":
        st.header("AI Authenticator")
        file = st.file_uploader("Upload Image", type=["png", "jpg"])
        if file:
            img = Image.open(file).convert('RGB').resize((64, 64))
            t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(img).unsqueeze(0).to(device)
            if st.button("Analyze"):
                score = disc(t).item()
                if score > 0.5: st.success(f"Prediction: REAL ({score*100:.1f}%)")
                else: st.error(f"Prediction: AI FAKE ({(1-score)*100:.1f}%)")

    elif page == "Gallery":
        st.header("Session History")
        if not st.session_state.history:
            st.info("No images generated yet.")
        else:
            cols = st.columns(4)
            for idx, img in enumerate(st.session_state.history):
                cols[idx % 4].image(img)
