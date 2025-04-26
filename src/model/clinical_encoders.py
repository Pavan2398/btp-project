# ecg_vqa_system/src/model/clinical_encoders.py
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from transformers import AutoModel

class MedicalImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b4', in_channels=1)
        self.lead_attention = nn.MultiheadAttention(embed_dim=1792, num_heads=8)
        
    def forward(self, x):
        x = self.base.extract_features(x)
        x = x.flatten(2).permute(2, 0, 1)
        attn_out, _ = self.lead_attention(x, x, x)
        return attn_out.mean(dim=0)

class ClinicalTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed")
        self.proj = nn.Linear(768, 512)
        
    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids)
        return self.proj(outputs.last_hidden_state[:, 0])