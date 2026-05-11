import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import io

# --- Page Config ---
st.set_page_config(page_title="AnimeGen AI Pro", page_icon="🎨", layout="wide")

# --- Load CSS ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

local_css("style.css")

# --- Model Architecture ---
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

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator()
    checkpoint = torch.load("best_gan_model.pth", map_location=device)
    
    # Fix: Extract only the Generator weights from the checkpoint dictionary
    if "modelG_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["modelG_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device).eval()
    return model, device

def generate_images(model, device, num_images=1, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn(num_images, 100, 1, 1, device=device)
    with torch.no_grad():
        fake = model(noise).detach().cpu()
    grid = make_grid(fake, padding=2, normalize=True)
    img_array = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_array)

# --- Sidebar ---
with st.sidebar:
    st.title("🚀 AnimeGen Pro")
    menu = st.radio("Navigation", ["Home", "AI Generator", "Analytics"])
    st.markdown("---")
    st.caption(f"Running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# --- Logic ---
net_model, device_type = load_model()

if menu == "Home":
    st.title("Welcome to the Future of Anime Art 🌌")
    st.write("Leveraging Deep Convolutional GANs to synthesize unique characters.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "DCGAN")
    c2.metric("Resolution", "64x64 px")
    c3.metric("Status", "Ready")

elif menu == "AI Generator":
    st.header("🎨 Neural Generator")
    col1, col2 = st.columns([1, 2])
    with col1:
        seed = st.number_input("Seed", value=42)
        if st.button("Generate ✨"):
            img = generate_images(net_model, device_type, 1, seed)
            st.session_state['last_img'] = img
    with col2:
        if 'last_img' in st.session_state:
            st.image(st.session_state['last_img'], use_column_width=True)
            buf = io.BytesIO()
            st.session_state['last_img'].save(buf, format="PNG")
            st.download_button("Download PNG", buf.getvalue(), "anime.png")

elif menu == "Analytics":
    st.header("📊 Model Technical Specs")
    st.code(str(net_model))
