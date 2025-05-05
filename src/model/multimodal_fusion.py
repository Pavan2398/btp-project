# ecg_vqa_system/src/model/multimodal_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# In src/model/multimodal_fusion.py
class MedicalCrossAttention(nn.Module):
    def __init__(self, img_dim=1792, txt_dim=512):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, 512) 
        self.txt_proj = nn.Linear(txt_dim, 512) 
        self.attention = nn.MultiheadAttention(512, 8)
        
    def forward(self, img_feats, txt_feats):
        projected_img = self.img_proj(img_feats)
        projected_txt = self.txt_proj(txt_feats)
        attn_output, _ = self.attention(
            projected_img.unsqueeze(1),
            projected_txt.unsqueeze(1),
            projected_txt.unsqueeze(1)
        )
        return attn_output.squeeze()



class DiagnosticGate(nn.Module):
    def __init__(self, input_dim=512):  
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256),  
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        return features * self.gate(features)
