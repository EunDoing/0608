
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torchvision.transforms as transforms
import torch

# 커스텀 테마 적용
with open("custom_streamlit_theme.css") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

st.title("내가 그린 은하는 어떤 은하?")
st.write("🎨 원하는 색으로 은하를 그려보세요!")

# 색상 선택 및 캔버스 생성
stroke_color = st.color_picker("선 색상 선택", "#ffffff")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.3)",
    stroke_width=6,
    stroke_color=stroke_color,
    background_color="#f0f0f0",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 이미지가 있다면 예측 처리
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data, caption="내가 그린 은하")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    img_tensor = transform(canvas_result.image_data.astype("uint8")[:, :, :3]).unsqueeze(0)

    # 더미 예측 (실제 모델 연결 가능)
    st.subheader("📊 예측 결과")
    st.write("이 은하는 **나선형(spiral)** 은하일 확률이 높습니다!")
