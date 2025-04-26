# ecg_vqa_system/src/engine/clinical_trainer.py
import torch
from tqdm import tqdm

class MedicalTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            self.optimizer.zero_grad()
            
            images = batch['image'].to(self.device)
            texts = batch['input_ids'].to(self.device)
            answers = batch['answer'].to(self.device)
            
            outputs = self.model(images, texts)
            loss = torch.nn.functional.nll_loss(outputs, answers)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)