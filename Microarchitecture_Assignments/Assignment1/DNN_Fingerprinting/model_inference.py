import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import argparse
import time

# Define available models
MODEL_MAP = {
    "alexnet": models.alexnet,
    "resnet18": models.resnet18,
    "vgg11": models.vgg11,
    "densenet": models.densenet121,
    "squeezenet": models.squeezenet1_0,
}

def load_model(model_name):
    """Load a pre-trained model."""
    if model_name not in MODEL_MAP:
        raise ValueError(f"Invalid model name. Choose from: {list(MODEL_MAP.keys())}")

    model = MODEL_MAP[model_name](weights="IMAGENET1K_V1")  # Use 'weights' instead of 'pretrained'
    model.eval()  # Set to evaluation mode
    return model

def preprocess_image():
    """Generate a random RGB image and apply transformations."""
    image = Image.effect_noise((224, 224), 100).convert("RGB")  # Convert grayscale to RGB
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def run_inference(model, image_tensor, num_iterations=1000):
    """Run inference for the specified number of iterations."""
    with torch.no_grad():
        for i in range(num_iterations):
            _ = model(image_tensor)

def main():
    parser = argparse.ArgumentParser(description="Run inference on a random image using a chosen model.")
    parser.add_argument("model", choices=MODEL_MAP.keys(), help="Choose a model for inference.")
    args = parser.parse_args()

    model = load_model(args.model)
    image_tensor = preprocess_image()

    print(f"Running inference 1000 times using {args.model}...")
    start_time = time.time()
    run_inference(model, image_tensor, 1000)
    end_time = time.time()
    
    print(f"Completed 1000 iterations in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
