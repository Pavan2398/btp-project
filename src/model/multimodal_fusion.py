# ecg_vqa_system/src/model/multimodal_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MedicalCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_proj = nn.Linear(1792, 512)
        self.txt_proj = nn.Linear(512, 512)
        self.attention = nn.MultiheadAttention(512, 8)
        
    def forward(self, img_feats, txt_feats):
        img = self.img_proj(img_feats)
        txt = self.txt_proj(txt_feats)
        attn_out, _ = self.attention(
            img.unsqueeze(1), 
            txt.unsqueeze(1), 
            txt.unsqueeze(1)
        )
        return attn_out.squeeze()

class DiagnosticGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        return features * self.gate(features)