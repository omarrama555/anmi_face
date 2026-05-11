import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image
import io
import os
import time
import random

# --- Architectures (Exact match with your notebook) ---
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

# --- App Configuration ---
st.set_page_config(page_title="AnimeGen Pro", layout="wide")

# Force directory for outputs
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Load Style
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen, disc = Generator(), Discriminator()
    
    # Path diagnostic
    model_name = "best_gan_model.pth"
    if not os.path.exists(model_name):
        return None, None, device, False
        
    try:
        ckpt = torch.load(model_name, map_location=device)
        # Handle different save formats
        g_state = ckpt["modelG_state_dict"] if "modelG_state_dict" in ckpt else ckpt
        d_state = ckpt["modelD_state_dict"] if "modelD_state_dict" in ckpt else None
        
        gen.load_state_dict(g_state)
        if d_state: disc.load_state_dict(d_state)
        
        return gen.to(device).eval(), disc.to(device).eval(), device, True
    except Exception as e:
        st.error(f"Load Error: {e}")
        return None, None, device, False

# --- Auth System ---
if 'auth' not in st.session_state: st.session_state.auth = False
if 'history' not in st.session_state: st.session_state.history = []

if not st.session_state.auth:
    st.title("Secure Access Control")
    u, p = st.text_input("Username"), st.text_input("Password", type="password")
    if st.button("Login"):
        if u == "admin" and p == "admin":
            st.session_state.auth = True
            st.rerun()
        else: st.error("Unauthorized")
else:
    gen, disc, device, is_loaded = load_assets()
    
    if not is_loaded:
        st.error(f"Critical: 'best_gan_model.pth' not detected. Found files: {os.listdir('.')}")
        st.stop()

    page = st.sidebar.radio("Navigation", ["Generator", "Batch Engine", "Vision Scanner", "Gallery"])
    if st.sidebar.button("Logout"):
        st.session_state.auth = False
        st.rerun()

    if page == "Generator":
        st.header("Neural Image Synthesis")
        c1, c2 = st.columns([1, 2])
        with c1:
            seed = st.number_input("Seed", value=random.randint(0, 9999))
            noise_val = st.slider("Latent Strength", 0.5, 2.0, 1.0)
            if st.button("Generate"):
                with st.spinner("Decoding Latent Space..."):
                    torch.manual_seed(seed)
                    noise = torch.randn(1, 100, 1, 1, device=device) * noise_val
                    with torch.no_grad():
                        raw = gen(noise).detach().cpu()[0]
                    # Post-process: De-normalize
                    img = Image.fromarray(((raw.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).astype('uint8'))
                    st.session_state.history.append(img)
                    st.toast("Success")
        
        with c2:
            if st.session_state.history:
                last = st.session_state.history[-1]
                st.image(last, width=400)
                buf = io.BytesIO()
                last.save(buf, format="PNG")
                st.download_button("Download PNG", buf.getvalue(), f"anime_{seed}.png")

    elif page == "Batch Engine":
        st.header("High-Volume Generation")
        num = st.select_slider("Image Count", options=[4, 8, 16])
        if st.button("Run Batch"):
            noise = torch.randn(num, 100, 1, 1, device=device)
            with torch.no_grad():
                fakes = gen(noise).detach().cpu()
            grid = utils.make_grid(fakes, nrow=4, normalize=True)
            st.image(Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype('uint8')))

    elif page == "Vision Scanner":
        st.header("Authenticity Detector")
        up = st.file_uploader("Upload Image", type=["png", "jpg"])
        if up:
            img = Image.open(up).convert('RGB').resize((64, 64))
            t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(img).unsqueeze(0).to(device)
            if st.button("Scan"):
                score = disc(t).item()
                if score > 0.5: st.success(f"REAL ({score*100:.1f}%)")
                else: st.error(f"FAKE ({(1-score)*100:.1f}%)")

    elif page == "Gallery":
        st.header("Session Archive")
        if not st.session_state.history: st.info("No data.")
        else:
            cols = st.columns(4)
            for i, img in enumerate(st.session_state.history):
                cols[i % 4].image(img)
