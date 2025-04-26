# ecg_vqa_system/src/inference/clinical_predictor.py
import torch
from PIL import Image

class ClinicalPredictor:
    def __init__(self, model_path, tokenizer, transforms, device='cuda'):
        self.model = torch.load(model_path).to(device)
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.device = device
        
    def predict(self, image_path, question):
        img = self._process_image(image_path)
        tokens = self.tokenizer(
            question,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model(img.unsqueeze(0), tokens['input_ids'])
            return 'yes' if torch.argmax(output) == 1 else 'no'
            
    def _process_image(self, path):
        return self.transforms(Image.open(path)).to(self.device)