import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

# --- 1. REDEFINE THE ARCHITECTURE ---
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
    weights_path = 'saved_models/hand_sign_cnn_weights.pth'
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Could not find model weights at `{weights_path}`. Make sure the file is inside the 'saved_models' folder!")
        return None

model = load_model()

# --- 3. MEDIAPIPE INITIALIZATION ---
mp_hands = mp.solutions.hands
hands_processor = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

# --- 4. PREPROCESSING PIPELINE (STATIC) ---
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),   
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- 5. PREDICTION & BACKGROUND ISOLATION LOGIC ---
def predict_and_display(image):
    st.image(image, caption='Original Input', width=300)
    
    # Convert PIL Image to OpenCV numpy array (RGB)
    img_array = np.array(image)
    h, w, c = img_array.shape
    
    # Process image with MediaPipe to find hand landmarks
    results = hands_processor.process(img_array)
    
    if results.multi_hand_landmarks:
        st.write("🎯 Hand detected! Isolating background...")
        
        # Create an all-black mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get the boundary points of the detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        points = []
        for landmark in hand_landmarks.landmark:
            cx,