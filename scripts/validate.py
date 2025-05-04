import sys
import os
sys.path.append(os.path.abspath('.'))


import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.data.clinical_loader import ClinicalECGDataset
from src.model.clinical_vqa import ClinicalVQAModel
from src.configs.clinical_config import ClinicalConfig

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def main():
    print("[INFO] Initializing configuration...")
    config = ClinicalConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Load test dataset
    print("[INFO] Loading test dataset...")
    test_dataset = ClinicalECGDataset(
        'clinical_data/test.json',
        'clinical_data/',
        tokenizer_name="emilyalsentzer/Bio_ClinicalBERT"
    )
    print(f"[INFO] Test dataset loaded with {len(test_dataset)} samples.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    print("[INFO] DataLoader prepared.")

    # Load model and weights
    print("[INFO] Initializing model...")
    model = ClinicalVQAModel().to(device)
    print("[INFO] Loading pretrained weights...")
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.eval()
    print("[INFO] Model ready for inference.")

    # Inference
    print("[INFO] Starting inference...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            print(f"[INFO] Processing batch {i+1}...")
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['answer'].to(device)

            outputs = model(images, input_ids)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("[INFO] Inference completed.")
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n🎯 Accuracy: {acc:.4f}")

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds))

    print("\n📉 Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # Optionally show confusion matrix plot
    class_names = [str(i) for i in np.unique(all_labels)]  # Or replace with actual class names
    plot_confusion_matrix(cm, class_names)

if __name__ == "__main__":
    main()