"""RunPod Serverless Handler for DINOv2 Embeddings."""

import runpod
import torch
import httpx
import os
from io import BytesIO
from PIL import Image

# Set torch hub cache directory
os.environ['TORCH_HOME'] = '/app/.cache/torch'

# Load model ONCE at startup (outside handler for performance)
print("Loading DINOv2 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # Load DINOv2 ViT-L/14 via torch hub
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
    model = model.to(device)
    model.eval()
    print("DINOv2 model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Try smaller model as fallback
    print("Trying smaller dinov2_vits14 model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
    model = model.to(device)
    model.eval()
    print("DINOv2 vits14 model loaded successfully!")

# Standard ImageNet normalization
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def handler(job):
    """
    Process embedding request.
    
    Input: {"image_url": "https://..."} or {"image_base64": "..."}
    Output: {"embedding": [floats], "dimension": int}
    """
    job_input = job["input"]
    
    try:
        # Get image from URL or base64
        if "image_url" in job_input:
            response = httpx.get(job_input["image_url"], timeout=30, follow_redirects=True)
            response.raise_for_status()
            image_bytes = response.content
        elif "image_base64" in job_input:
            import base64
            image_bytes = base64.b64decode(job_input["image_base64"])
        else:
            return {"error": "Must provide image_url or image_base64"}
        
        # Process image
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = model(input_tensor)
        
        # Normalize and convert to list
        embedding = embedding.squeeze().cpu().numpy()
        embedding = embedding / (embedding.dot(embedding) ** 0.5)  # L2 normalize
        
        return {
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
        }
        
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# Start serverless handler
runpod.serverless.start({"handler": handler})
