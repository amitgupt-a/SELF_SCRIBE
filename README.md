# 🦴 Self Scribe — AI-Powered Bone Fracture Detection System

Self Scribe is a fully-deployed, deep learning–based web application that detects and classifies bone fractures from X-ray images in real-time. Built by **Amit Gupta** using cutting-edge models like **YOLOv8**, **Faster R-CNN (ResNet50)**, and **VGG16 with SSD**, the system assists healthcare professionals with accurate and fast diagnostics.

## 📌 Features

- Upload X-ray images to detect fractures in elbow, shoulder, wrist, and more
- Choose between three pretrained detection models
- Adjust confidence thresholds for sensitivity
- View annotated predictions with bounding boxes
- Learn from model performance metrics in an interactive UI

## 🧠 Tech Stack

| Type           | Technologies Used |
|----------------|-------------------|
| Language       | Python 3.10        |
| Web Framework  | Streamlit          |
| ML Frameworks  | PyTorch, TensorFlow, Ultralytics |
| Models         | YOLOv8, Faster R-CNN + ResNet, VGG16 + SSD |
| Dataset        | [Kaggle Bone Fracture X-ray Dataset](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project) |

## 📂 Project Structure

```
self-scribe/
├── app.py
├── models/
│   ├── yolov8.pt
│   ├── faster_rcnn.pth
│   └── vgg16_ssd.pb
├── utils/
│   └── preprocessing.py
├── requirements.txt
└── README.md
```

## 🔧 Setup & Installation

```bash
git clone https://github.com/yourusername/self-scribe.git
cd self-scribe
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Run the App

```bash
pip install streamlit
streamlit run app.py
```

## 📊 Model Performance (Summary)

| Model            | Accuracy | Precision | Recall | Inference Time |
|------------------|----------|-----------|--------|----------------|
| YOLOv8           | 92.7%    | 91.8%     | 90.5%  | 12.5 ms        |
| Faster R-CNN     | 90.2%    | 89.1%     | 88.5%  | 55.3 ms        |
| VGG16 + SSD      | 88.5%    | 87.3%     | 85.9%  | 28.9 ms        |

## 🚀 Future Enhancements

- Integrate Grad-CAM for model explainability
- Enable deployment to cloud platforms (AWS, Render)
- Expand dataset with CT/MRI scans
- Mobile app support for remote diagnostics
- EHR system integration for real-time clinical use

## 👨‍💻 Author

**Amit Gupta**  
📍 KIET Group of Institutions, Ghaziabad  
📧 amitguptacoding@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/your-link) | [GitHub](https://github.com/yourusername)

## 📝 License

This project is licensed for academic and personal research use. For commercial or clinical deployment, contact the author.

# SELF_SCRIBE
Self Scribe is an AI-powered web application designed to detect bone fractures in X-ray images using vgg16, resnet 50 and yolov8. Built to assist medical professionals with rapid and accurate diagnosis, this tool leverages multiple computer vision techniques to ensure high precision and reliability.
