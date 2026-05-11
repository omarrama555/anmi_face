import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image
import io
import os

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
    def forward(self, input): return self.main(input)

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
    def forward(self, input): return self.main(input)

# --- Configuration & Resource Loading ---
st.set_page_config(page_title="AnimeGen Pro", layout="wide")

def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

@st.cache_resource
def init_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen, disc = Generator(), Discriminator()
    ckpt = torch.load("best_gan_model.pth", map_location=device)
    gen.load_state_dict(ckpt["modelG_state_dict"])
    disc.load_state_dict(ckpt["modelD_state_dict"])
    return gen.to(device).eval(), disc.to(device).eval(), device

# --- Session Management ---
if 'auth' not in st.session_state: st.session_state.auth = False
if 'history' not in st.session_state: st.session_state.history = []

if not st.session_state.auth:
    st.title("Secure Access")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "admin" and pwd == "admin":
            st.session_state.auth = True
            st.rerun()
else:
    gen, disc, device = init_models()
    page = st.sidebar.radio("Navigation", ["Generator", "Vision Scanner", "History", "System"])
    if st.sidebar.button("Logout"):
        st.session_state.auth = False
        st.rerun()

    if page == "Generator":
        st.header("Image Synthesis Engine")
        batch_size = st.slider("Batch Size", 1, 16, 1)
        seed = st.number_input("Seed", value=6969)
        if st.button("Generate"):
            torch.manual_seed(seed)
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            with torch.no_grad():
                fakes = gen(noise).detach().cpu()
            grid = utils.make_grid(fakes, padding=2, normalize=True)
            res_img = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype('uint8'))
            st.image(res_img)
            st.session_state.history.append(res_img)

    elif page == "Vision Scanner":
        st.header("Authentication Analysis")
        file = st.file_uploader("Upload Image", type=["png", "jpg"])
        if file:
            img = Image.open(file).convert('RGB').resize((64, 64))
            tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(img).unsqueeze(0).to(device)
            if st.button("Analyze"):
                score = disc(tensor).item()
                if score > 0.5: st.success(f"REAL: {score*100:.2f}%")
                else: st.error(f"FAKE: {(1-score)*100:.2f}%")

    elif page == "History":
        st.header("Session History")
        for i, img in enumerate(st.session_state.history):
            st.image(img, caption=f"Generation {i+1}")

    elif page == "System":
        st.header("System Metrics")
        st.write(f"Device: {device}")
        st.write(f"Total Cached Images: {len(st.session_state.history)}")
