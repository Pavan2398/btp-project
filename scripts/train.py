import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from src.data.clinical_loader import ClinicalECGDataset
from src.model import MedicalImageEncoder, ClinicalTextEncoder, MedicalCrossAttention, ClinicalClassifier
from src.engine.clinical_trainer import MedicalTrainer
from src.model.clinical_vqa import ClinicalVQAModel
from src.configs.clinical_config import ClinicalConfig


def main():
    config = ClinicalConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config.batch_size = 4
    config.image_encoder = 'efficientnet-b0'
    config.use_amp = True  

    
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

    model = ClinicalVQAModel().to(device)

    optimizer = torch.optim.AdamW([
        {'params': model.image_encoder.parameters(), 'lr': 1e-5},
        {'params': model.text_encoder.parameters(), 'lr': 2e-5},
        {'params': model.cross_attn.parameters(), 'lr': 3e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-5)

    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

   
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_ckpt = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    start_epoch = 0

    if os.path.exists(latest_ckpt):
        print(f"🔁 Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed at epoch {start_epoch}")

  
    trainer = MedicalTrainer(model, train_loader, None, optimizer, device, scaler=scaler)

  
    for epoch in range(start_epoch, config.epochs):
        loss, accuracy = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}%")

        
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }, ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

     
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }, latest_ckpt)

    torch.save(model.state_dict(), config.model_save_path)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
