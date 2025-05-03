import sys
import os
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

    # Modify training parameters
    config.batch_size = 4  
    config.image_encoder = 'efficientnet-b0'
    config.use_amp = True

    checkpoint_path = "checkpoint.pth"

    # Dataset and DataLoader
    train_dataset = ClinicalECGDataset(
        'clinical_data/train.json',
        'clinical_data/',
        tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # Model and optimizer
    model = ClinicalVQAModel().to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.image_encoder.parameters(), 'lr': 1e-5},
        {'params': model.text_encoder.parameters(), 'lr': 2e-5},
        {'params': model.cross_attn.parameters(), 'lr': 3e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-5)

    start_epoch = 0

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    trainer = MedicalTrainer(model, train_loader, None, optimizer, device)

    # Training loop
    for epoch in range(start_epoch, config.epochs):
        loss, accuracy = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}%")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")

    # Final model save
    torch.save(model.state_dict(), config.model_save_path)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

if _name_ == "_main_":
    main()