import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# --- Page Config ---
st.set_page_config(page_title="AnimeGen Pro AI", page_icon="🕵️‍♀️", layout="wide")

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
    
    # Load weights
    gen.load_state_dict(checkpoint["modelG_state_dict"])
    disc.load_state_dict(checkpoint["modelD_state_dict"])
    
    gen.to(device).eval()
    disc.to(device).eval()
    return gen, disc, device

gen, disc, device = load_models()

# --- Sidebar Navigation ---
st.sidebar.title("🚀 AnimeGen Pro")
page = st.sidebar.radio("Navigation", ["Home", "AI Generator", "Vision Scanner (Fake vs Real)"])

if page == "Home":
    st.title("Dual-Core Anime AI 🌌")
    st.write("Generate high-quality anime faces or use the Discriminator to detect AI-generated art.")

elif page == "AI Generator":
    st.header("🎨 Image Synthesis")
    seed = st.number_input("Seed", value=42)
    if st.button("Generate ✨"):
        noise = torch.randn(1, 100, 1, 1, device=device)
        with torch.no_grad():
            img_tensor = gen(noise).detach().cpu()[0]
        # De-normalize and convert to PIL
        img_array = ((img_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).astype('uint8')
        st.image(Image.fromarray(img_array), width=300)

elif page == "Vision Scanner (Fake vs Real)":
    st.header("🕵️‍♀️ AI Authenticator")
    st.write("Upload an image (64x64 preferred) to check if it's Real or AI-Generated.")
    
    uploaded_file = st.file_uploader("Choose an anime face image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB').resize((64, 64))
        st.image(img, caption="Target Image", width=200)
        
        # Preprocess
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        if st.button("Analyze Image"):
            with torch.no_grad():
                output = disc(img_tensor).item()
            
            # Confidence Logic (Close to 1 is Real, Close to 0 is Fake)
            st.subheader("Results:")
            if output > 0.5:
                st.success(f"Prediction: REAL (Confidence: {output*100:.2f}%)")
            else:
                st.error(f"Prediction: FAKE/AI-GENERATED (Confidence: {(1-output)*100:.2f}%)")
            
            st.progress(output)
