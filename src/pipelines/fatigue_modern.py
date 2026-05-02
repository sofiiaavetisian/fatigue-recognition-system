from __future__ import annotations
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
from collections import deque

class ModernFatigueDetector:
    def __init__(self, config: dict):
        # 1. Configuration & Smoothing Setup
        self.cfg = config.get('modern', config) 
        self.threshold = self.cfg.get('threshold', 0.5)
        
        # Buffer for stability (averages the last 5 frames)
        window_size = self.cfg.get('smoothing_window', 5)
        self.score_buffer = deque(maxlen=window_size)
        
        # 2. Device Selection
        # MANDATORY: Using 'cpu' because AMD GPUs on Intel Macs crash with 'mps'
        self.device = torch.device("cpu")
        print("Modern AI: Using CPU for stability on Intel/AMD Mac.")
        
        # 3. Image Preprocessing
        # Added CenterCrop to prevent face-squashing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
        
        # 4. Initialize Architecture
        self.model = models.mobilenet_v3_small(weights=None)
        num_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_features, 1)
        
        # 5. Load the Trained Weights
        MODEL_PATH = self.cfg.get('model_path', 'models/fatigue_model.pt')
        if os.path.exists(MODEL_PATH):
            try:
                state_dict = torch.load(MODEL_PATH, map_location=self.device)
                if isinstance(state_dict, torch.nn.Module):
                    state_dict = state_dict.state_dict()
                
                self.model.load_state_dict(state_dict)
                print(f"Successfully loaded AI model: {MODEL_PATH}")
            except Exception as e:
                print(f"Error loading model weights: {e}")
        else:
            print(f"WARNING: No model found at {MODEL_PATH}")
        
        self.model.to(self.device)
        self.model.eval() # Sets model to evaluation mode

    def analyze(self, frame: np.ndarray) -> float:
        if frame is None: return 0.0
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_t = self.transform(rgb_frame).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(img_t)
                raw_val = logits.item()
                
                sensitivity_offset = 8.0 
                boosted_logit = raw_val + sensitivity_offset
                probability = torch.sigmoid(torch.tensor(boosted_logit)).item()
            
            # print(f"AI: Raw={raw_val:.1f} | Boosted={boosted_logit:.1f} | Prob={probability:.2f}")

            self.score_buffer.append(probability)
            return sum(self.score_buffer) / len(self.score_buffer)
        except Exception as e:
            return 0.0

# Singleton pattern for the app
_detector = None

def modern_fatigue_score(frame: np.ndarray, settings: dict = None) -> float:
    global _detector
    if _detector is None and settings is not None:
        _detector = ModernFatigueDetector(settings)
    
    if _detector:
        return _detector.analyze(frame)
    return 0.0