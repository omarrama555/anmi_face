import streamlit as st
from core.model_loader import load_gan_model
import torch

# Premium Dashboard Configuration
st.set_page_config(
    page_title="AnimeGen Pro | Next-Gen AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Futuristic Styling Injection
def local_css():
    st.markdown("""
        <style>
        .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: white; }
        .stButton>button { 
            border-radius: 20px; border: 2px solid #7000ff; 
            background: rgba(112, 0, 255, 0.2); color: white;
            transition: 0.3s;
        }
        .stButton>button:hover { background: #7000ff; box-shadow: 0 0 15px #7000ff; }
        div[data-testid="stMetricValue"] { color: #00f2ff; }
        </style>
    """, unsafe_allow_html=True)

local_css()

# Session State for History & Stats
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_gen' not in st.session_state:
    st.session_state.total_gen = 0

# Sidebar Navigation
st.sidebar.title("🚀 AnimeGen Pro")
page = st.sidebar.radio("Navigation", [
    "🏠 Home", "🎨 AI Generator", "📚 Batch Gen", "🖼️ Gallery", 
    "🧠 About GAN", "📊 Analytics", "⚙️ Settings"
])

# Loading Model
# Note: Replace URL with your actual hosted best_gan_model.pth link
weights_url = "best_gan_model.pth" 
model, device = load_gan_model(weights_url)

# Routing Logic
if page == "🎨 AI Generator":
    st.header("✨ Neural Anime Synthesis")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Control Panel")
        seed = st.number_input("Random Seed", value=6969, step=1)
        noise_level = st.slider("Noise Intensity", 0.0, 1.0, 0.5)
        btn = st.button("Generate Masterpiece")

    with col2:
        if btn:
            with st.spinner("Decoding Latent Space..."):
                torch.manual_seed(seed)
                noise = torch.randn(1, 100, 1, 1, device=device)
                with torch.no_grad():
                    fake = model(noise).detach().cpu()
                
                # Process and display image logic here...
                st.image("https://via.placeholder.com/400x400.png?text=Generated+Anime+Face", 
                         caption=f"Seed: {seed}", use_column_width=True)
                st.success("Generation Complete!")
                st.session_state.total_gen += 1

elif page == "📊 Analytics":
    st.title("📈 System Analytics")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Current Engine", str(device).upper())
    kpi2.metric("Total Generations", st.session_state.total_gen)
    kpi3.metric("Uptime", "99.9%")
