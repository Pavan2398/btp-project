# ecg_vqa_system/src/engine/clinical_trainer.py

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # ✅ Correct AMP import

class MedicalTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.scaler = GradScaler()  # ✅ Removed device_type
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            self.optimizer.zero_grad()

            images = batch['image'].to(self.device)
            texts = batch['input_ids'].to(self.device)
            answers = batch['answer'].to(self.device)

            with autocast():
                outputs = self.model(images, texts)
                loss = self.loss_fn(outputs, answers)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_correct += (preds == answers).sum().item()
            total_samples += answers.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy
