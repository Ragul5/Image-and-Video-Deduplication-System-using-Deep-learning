"""
CNN Models for Image Deduplication
Implements ResNet-152, ConvNeXt, and Swin Transformer models for feature extraction
"""

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import swin_b, Swin_B_Weights

# Set device to CPU for compatibility
device = torch.device("cpu")

# Singleton pattern for model loading (to avoid loading multiple times)
_resnet_model = None
_convnext_model = None
_swin_model = None

def get_resnet_model():
    """Get ResNet-152 model instance."""
    global _resnet_model
    if _resnet_model is None:
        # Load pre-trained ResNet-152
        weights = ResNet152_Weights.DEFAULT
        model = resnet152(weights=weights)
        model.eval()  # Set to evaluation mode
        
        # Remove the classification layer
        _resnet_model = torch.nn.Sequential(*list(model.children())[:-1])
        _resnet_model.to(device)
    
    return _resnet_model

def get_convnext_model():
    """Get ConvNeXt model instance."""
    global _convnext_model
    if _convnext_model is None:
        # Load pre-trained ConvNeXt
        weights = ConvNeXt_Base_Weights.DEFAULT
        model = convnext_base(weights=weights)
        model.eval()  # Set to evaluation mode
        
        # Remove the classification layer
        _convnext_model = torch.nn.Sequential(*list(model.children())[:-1])
        _convnext_model.to(device)
    
    return _convnext_model

def get_swin_model():
    """Get Swin Transformer model instance."""
    global _swin_model
    if _swin_model is None:
        # Load pre-trained Swin Transformer
        weights = Swin_B_Weights.DEFAULT
        model = swin_b(weights=weights)
        model.eval()  # Set to evaluation mode
        
        # Remove the classification layer
        _swin_model = torch.nn.Sequential(*list(model.children())[:-1])
        _swin_model.to(device)
    
    return _swin_model

def preprocess_image(image_path):
    """Preprocess an image for CNN feature extraction."""
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        
        # Define transformations
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Apply transformations
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def extract_features_resnet(image_path):
    """Extract features using ResNet-152."""
    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        return None
    
    with torch.no_grad():
        model = get_resnet_model()
        features = model(image_tensor)
        features = features.squeeze().cpu().numpy()
        
        # Normalize features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
    return features

def extract_features_convnext(image_path):
    """Extract features using ConvNeXt."""
    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        return None
    
    with torch.no_grad():
        model = get_convnext_model()
        features = model(image_tensor)
        features = features.squeeze().cpu().numpy()
        
        # Normalize features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
    return features

def extract_features_swin(image_path):
    """Extract features using Swin Transformer."""
    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        return None
    
    with torch.no_grad():
        model = get_swin_model()
        features = model(image_tensor)
        features = features.squeeze().cpu().numpy()
        
        # Normalize features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
    return features

def compute_similarity(features1, features2):
    """Compute cosine similarity between two feature vectors."""
    return np.dot(features1, features2)

def are_similar_cnn(features1, features2, threshold=0.9):
    """Check if two images are similar based on CNN features."""
    similarity = compute_similarity(features1, features2)
    return similarity >= threshold