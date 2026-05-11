import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import io
import time

# --- إعدادات الصفحة والهوية البصرية ---
st.set_page_config(
    page_title="AnimeGen AI Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- استايل Glassmorphism و Dark Theme ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stApp {
        background: linear-gradient(160deg, #0e1117 0%, #161b22 100%);
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(45deg, #7928CA, #FF0080);
        color: white;
        border: none;
        padding: 10px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(255, 0, 128, 0.4);
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(180deg, #161b22, #0e1117);
    }
    </style>
    """, unsafe_allow_html=True)

# --- تعريف بنية مودل Generator (بناءً على الـ Notebook الخاص بك) ---
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

# --- دوال المساعدة ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator()
    try:
        # تحميل المودل من الملف المرفق
        state_dict = torch.load("best_gan_model.pth", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

def generate_images(model, device, num_images=1, seed=None, noise_val=100):
    if seed is not None:
        torch.manual_seed(seed)
    
    noise = torch.randn(num_images, noise_val, 1, 1, device=device)
    with torch.no_grad():
        fake_images = model(noise).detach().cpu()
    
    # تحويل الصور من Tensor إلى PIL
    grid = make_grid(fake_images, padding=2, normalize=True)
    img_array = grid.permute(1, 2, 0).numpy()
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

# --- إدارة حالة الجلسة (Session State) ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- القائمة الجانبية (Sidebar Navigation) ---
with st.sidebar:
    st.title("🚀 AnimeGen Pro")
    st.markdown("---")
    menu = st.radio("القائمة الرئيسية", 
        ["🏠 Home", "🎨 AI Generator", "📚 Batch Generation", "📊 Analytics", "⚙️ Settings"])
    
    st.markdown("---")
    st.info(f"المنصة تعمل الآن على: **{'GPU' if torch.cuda.is_available() else 'CPU'}**")

# --- الصفحات ---
net_model, device_type = load_model()

if menu == "🏠 Home":
    st.title("مرحباً بك في مستقبل فن الأنمي 🌌")
    st.write("استخدم قوة الـ Generative Adversarial Networks لإنشاء شخصيات أنمي فريدة في ثوانٍ.")
    col1, col2, col3 = st.columns(3)
    col1.metric("المودل", "DCGAN")
    col2.metric("الدقة", "64x64 px")
    col3.metric("الحالة", "جاهز")
    
    st.image("https://raw.githubusercontent.com/pytorch/hub/master/images/dcgan_generator.png", caption="بنية الشبكة المستخدمة")

elif menu == "🎨 AI Generator":
    st.header("🎨 لوحة توليد الصور الذكية")
    
    col_ctrl, col_res = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("إعدادات التحكم")
        seed_input = st.number_input("الرقم السري (Seed)", value=42, step=1)
        use_random = st.checkbox("استخدام عشوائي بالكامل")
        noise_slider = st.select_slider("مستوى التفاصيل (Latent Dim)", options=[100], value=100)
        
        generate_btn = st.button("توليد الصورة الآن ✨")
    
    with col_res:
        if generate_btn:
            with st.spinner("جاري معالجة المصفوفات العصبية..."):
                final_seed = None if use_random else seed_input
                img = generate_images(net_model, device_type, num_images=1, seed=final_seed)
                st.image(img, use_column_width=True, caption="الصورة المولدة بواسطة AI")
                
                # حفظ في التاريخ
                st.session_state.history.append(img)
                
                # زر التحميل
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.download_button("تحميل الصورة 📥", buf.getvalue(), "anime_face.png", "image/png")

elif menu == "📚 Batch Generation":
    st.header("📚 توليد دفعات متعددة")
    num_batch = st.slider("عدد الصور", 4, 32, 16, step=4)
    
    if st.button(f"توليد {num_batch} صورة دفعة واحدة"):
        with st.spinner("جاري إنشاء الشبكة الرسومية..."):
            img_batch = generate_images(net_model, device_type, num_images=num_batch)
            st.image(img_batch, use_column_width=True)
            
            buf = io.BytesIO()
            img_batch.save(buf, format="PNG")
            st.download_button("تحميل الشبكة كاملة 📥", buf.getvalue(), "batch_anime.png")

elif menu == "📊 Analytics":
    st.header("📊 لوحة البيانات الفنية")
    st.write("تفاصيل المودل `best_gan_model.pth`:")
    st.code(str(net_model))
    
    # إحصائيات وهمية للمظهر الاحترافي
    c1, c2 = st.columns(2)
    c1.info(f"جهاز التشغيل: {device_type}")
    c2.success(f"عدد الصور المولدة في هذه الجلسة: {len(st.session_state.history)}")

# --- تذييل الصفحة ---
st.sidebar.markdown("---")
st.sidebar.caption("Senior AI Project | 2024 ©")
