import sys
import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(_file_), '..')))

from src.model.clinical_vqa import ClinicalVQAModel
from src.configs.clinical_config import ClinicalConfig


LABELS = {
    0: 'Yes',
    1: 'No'
}

def load_ecg_image(image_path):
    """Preprocess the 512x512 ECG image."""
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])
    image = Image.open(image_path).convert("L")  
    return transform(image).unsqueeze(0)  

def run_inference(image_path, question):
    config = ClinicalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading model...")
    model = ClinicalVQAModel().to(device)
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.eval()

    print("[INFO] Tokenizing question...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    encoded = tokenizer(question, return_tensors='pt', truncation=True, padding=True, max_length=128)
    input_ids = encoded['input_ids'].to(device)

    print("[INFO] Loading and preprocessing image...")
    image_tensor = load_ecg_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(image_tensor, input_ids)
        print("Model output shape:", outputs.shape)
        predicted_idx = torch.argmax(outputs).item()
        predicted_label = LABELS.get(predicted_idx, str(predicted_idx))

    print(f"\n Predicted: {predicted_label} (Class ID: {predicted_idx})")
    return predicted_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ECG-VQA inference.")
    parser.add_argument("image_path", type=str, help="Path to the ECG image")
    parser.add_argument("question", type=str, help="Question about the ECG")
    args = parser.parse_args()

    run_inference(args.image_path, args.question)