from __future__ import annotations
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

class ModernFatigueDetector:
    def __init__(self, config: dict):
        # 1. Configuration & Thresholds
        # Expects config['modern'] or a flat dict with thresholds
        self.cfg = config.get('modern', config) 
        self.threshold = self.cfg.get('threshold', 0.5)
        
        # 2. Device Selection (Use 'mps' for Mac M1/M2/M3, else 'cpu')
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # 3. Image Preprocessing (Standard AI normalization)
        # Resizes to 224x224 and normalizes colors to match the AI's training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 4. Load Pre-trained Model (MobileNetV3 Small)
        # We modify the final 'classifier' layer to output a single score
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        num_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_features, 1)
        
        MODEL_PATH = 'models/fatigue_model.pt'
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            print("Successfully loaded trained fatigue model!")
        else:
            print("WARNING: No trained model found at models/fatigue_model.pt. Using untrained weights.")
        
        self.model.to(self.device)
        self.model.eval()

    def analyze(self, frame: np.ndarray) -> float:
        """Runs the AI model on a single frame and returns a probability score."""
        try:
            # Convert OpenCV (BGR) to PIL (RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            
            # Apply transforms and add batch dimension
            img_t = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Forward pass through the network
                output = self.model(img_t)
                # Sigmoid turns the raw number into a 0.0 to 1.0 probability
                probability = torch.sigmoid(output).item()
                
            return probability
        except Exception as e:
            print(f"Modern Detector Error: {e}")
            return 0.0

# Singleton pattern to prevent reloading the model every frame
_detector = None

def modern_fatigue_score(frame: np.ndarray, settings: dict = None) -> float:
    """Entry point for the live app to call the Modern AI brain."""
    global _detector
    if _detector is None and settings is not None:
        _detector = ModernFatigueDetector(settings)
    
    if _detector:
        return _detector.analyze(frame)
    return 0.0