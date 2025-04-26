# ecg_vqa_system/scripts/train.py
import torch
from src.data.clinical_loader import ClinicalECGDataset
from src.model import MedicalImageEncoder, ClinicalTextEncoder, MedicalCrossAttention, ClinicalClassifier
from src.engine.clinical_trainer import MedicalTrainer
from torch.utils.data import DataLoader

def main():
    config = ClinicalConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize components
    train_dataset = ClinicalECGDataset('clinical_data/train.json', 'clinical_data/ecg_images', config.tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    model = torch.nn.Sequential(
        MedicalImageEncoder(),
        ClinicalTextEncoder(),
        MedicalCrossAttention(),
        ClinicalClassifier()
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    trainer = MedicalTrainer(model, train_loader, None, optimizer, device)
    
    for epoch in range(config.epochs):
        loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} Loss: {loss:.4f}")
    
    torch.save(model.state_dict(), config.model_save_path)

if __name__ == "__main__":
    main()