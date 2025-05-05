# ecg_vqa_system/src/model/clinical_heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClinicalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),            
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, x):
        return self.head(x)
