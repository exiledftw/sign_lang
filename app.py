import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

# --- 1. REDEFINE THE ARCHITECTURE ---
# Must match the training file exactly
class HandSignCNN(nn.Module):
    def __init__(self):
        super(HandSignCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 2. LOAD TRAINED MODEL WITH CACHING ---
@st.cache_resource
def load_model():
    model = HandSignCNN()
    # Reverted back to your preferred subdirectory path
    weights_path = 'saved_models/hand_sign_cnn_weights.pth'
    
    try:
        # Load to CPU to ensure compatibility across web hosting platforms
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Could not find model weights at `{weights_path}`. Make sure the file is inside the 'saved_models' folder!")
        return None

model = load_model()
# --- 3. GLOBAL PREPROCESSING (STATIC) ---
# Removed CenterCrop from here so it doesn't crash on startup
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),   
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- 4. PREDICTION LOGIC ---
def predict_and_display(image):
    st.image(image, caption='Image for Prediction', width=300)
    st.write("🔄 Running inference...")
    
    # 1. Dynamically crop a perfect square from the center of the image to fix distortion
    cropped_image = F.center_crop(image, min(image.size))
    
    # 2. Resize and normalize the square matrix
    input_tensor = preprocess(cropped_image)       
    input_batch = input_tensor.unsqueeze(0) # Add batch dimension -> [1, 3, 64, 64]

    if model is not None:
        with torch.no_grad():
            outputs = model(input_batch)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        st.success(f"### Predicted Digit: **{predicted_idx.item()}**")
        st.info(f"Confidence Level: **{confidence.item() * 100:.2f}%**")

# --- 5. STREAMLIT INTERFACE UI ---
st.title("🖐️ Live Hand Sign Predictor")
st.write("Take a picture with your webcam or upload an image to predict the hand gesture (0-9).")

# Create tabs for a cleaner user experience
tab1, tab2 = st.tabs(["📸 Webcam Input", "📁 Upload Image"])

# Tab 1: Webcam
with tab1:
    st.write("Center your hand in the frame and click to capture.")
    camera_image = st.camera_input("Capture Hand Sign")
    
    if camera_image is not None:
        image = Image.open(camera_image).convert('RGB')
        predict_and_display(image)

# Tab 2: File Upload
with tab2:
    uploaded_file = st.file_uploader("Choose a hand sign image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        predict_and_display(image)