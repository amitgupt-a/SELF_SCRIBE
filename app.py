import streamlit as st  
from ultralytics import YOLO
import random
import PIL
from PIL import Image, ImageOps
import numpy as np
import torchvision
import torch

from sidebar import Sidebar
import rcnnres, vgg
import warnings
warnings.filterwarnings("ignore")  # hide deprecation warnings

# Sidebar 
sb = Sidebar()

title_image = sb.title_img
model = sb.model_name
conf_threshold = sb.confidence_threshold

# Main Page
st.title("Self Scribe")
st.write("The Application provides Bone Fracture Detection using multiple state-of-the-art computer vision models such as Yolo V8, ResNet, and CNN. ")

# CSS Styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 2px 2px 2px 2px;
        gap: 8px;
        padding-left: 10px;
        padding-right: 10px;
        padding-top: 8px;
        padding-bottom: 8px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #7f91ad;
    }
</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Overview", "Test"])

with tab1:
    st.markdown("### Overview")
    st.markdown("This application performs bone fracture detection using state-of-the-art computer vision models.")

# Model Path
yolo_path = "weights/yolov8.pt"  # Correct path for macOS/Linux

# Test Tab (where image upload and model inference happens)
with tab2:
    st.markdown("### Upload & Test")

    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def set_clicked():
        st.session_state.clicked = True

    st.button('Upload Image', on_click=set_clicked)

    if st.session_state.clicked:
        image = st.file_uploader("", type=["jpg", "png"])

        if image is not None:
            st.write("You selected the file:", image.name)

            if model == 'YoloV8':
                try:
                    yolo_detection_model = YOLO(yolo_path)
                    yolo_detection_model.load()
                except Exception as ex:
                    st.error(f"Unable to load model. Check the specified path: {yolo_path}")
                    st.error(ex)

                col1, col2 = st.columns(2)

                with col1:
                    uploaded_image = PIL.Image.open(image)
                    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

                    if st.button("Execution"):
                        with st.spinner("Running..."):
                            try:
                                res = yolo_detection_model.predict(uploaded_image, conf=conf_threshold, augment=True, max_det=1)
                                boxes = res[0].boxes
                                res_plotted = res[0].plot()[:, :, ::-1]

                                if len(boxes) == 1:
                                    names = yolo_detection_model.names
                                    probs = boxes.conf[0].item()

                                    for r in res:
                                        for c in r.boxes.cls:
                                            pred_class_label = names[int(c)]

                                    with col2:
                                        st.image(res_plotted, caption="Detected Image", use_column_width=True)
                                        with st.expander("Detection Results"):
                                            for box in boxes:
                                                st.write(pred_class_label)
                                                st.write(probs)
                                                st.write(box.xywh)
                                else:
                                    with col2:
                                        st.image(res_plotted, caption="Detected Image", use_column_width=True)
                                        with st.expander("Detection Results"):
                                            st.write("No Detection")
                            except Exception as ex:
                                st.write("Error during prediction.")
                                st.write(ex)

            elif model == 'VGG16':
                vgg_model = vgg.get_vgg_model()
                device = torch.device('cpu')
                vgg_model.to(device)

                col1, col2 = st.columns(2)

                with col1:
                    uploaded_image = PIL.Image.open(image)
                    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

                    content = Image.open(image).convert("RGB")
                    to_tensor = torchvision.transforms.ToTensor()
                    content = to_tensor(content).unsqueeze(0)

                    if st.button("Execution"):
                        with st.spinner("Running..."):
                            output = rcnnres.make_prediction(vgg_model, content, conf_threshold)
                            fig, _ax, class_name = rcnnres.plot_image_from_output(content[0].detach(), output[0])

                            with col2:
                                st.image(rcnnres.figure_to_array(fig), caption="Detected Image", use_column_width=True)
                                with st.expander("Detection Results"):
                                    st.write(class_name)
                                    st.write(output)

            elif model == 'FastRCNN with ResNet':
                f_model = rcnnres.get_model()
                f_model.eval()

                image_pil = Image.open(image).convert("RGB")
                transform = torchvision.transforms.ToTensor()
                img_tensor = transform(image_pil).unsqueeze(0)

                col1, col2 = st.columns(2)

                with col1:
                    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

                if st.button("Execution"):
                    with st.spinner("Running..."):
                        preds = rcnnres.make_prediction(f_model, img_tensor, conf_threshold)
                        fig, ax, label = rcnnres.plot_image_from_output(img_tensor[0], preds[0])

                        with col2:
                            st.pyplot(fig)
                            with st.expander("Detection Results"):
                                st.write(label if label else "No fracture detected")
                                st.write(preds)


        else:
            st.write("Please upload an image to test")
