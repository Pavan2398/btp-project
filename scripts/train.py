import sys
import os

# Fix the import path problem
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from src.data.clinical_loader import ClinicalECGDataset
from src.model import MedicalImageEncoder, ClinicalTextEncoder, MedicalCrossAttention, ClinicalClassifier
from src.engine.clinical_trainer import MedicalTrainer
from src.model.clinical_vqa import ClinicalVQAModel
from src.configs.clinical_config import ClinicalConfig


def main():
    config = ClinicalConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reduce batch size for limited GPU memory (e.g., Colab)
    config.batch_size = 4  
    config.image_encoder = 'efficientnet-b0'  # Lighter encoder
    config.use_amp = True  # Enable mixed precision training

    # Dataset and DataLoader
    train_dataset = ClinicalECGDataset(
        'clinical_data/train.json',
        'clinical_data/',
        tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
    )
    # train_dataset = torch.utils.data.Subset(train_dataset, range(0, 4000))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # Model
    model = ClinicalVQAModel().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.image_encoder.parameters(), 'lr': 1e-5},
        {'params': model.text_encoder.parameters(), 'lr': 2e-5},
        {'params': model.cross_attn.parameters(), 'lr': 3e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-5)

    # Trainer
    trainer = MedicalTrainer(model, train_loader, None, optimizer, device)

    # Training loop
    for epoch in range(config.epochs):
        loss, accuracy = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}%")

    torch.save(model.state_dict(), config.model_save_path)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
