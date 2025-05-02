# ecg_vqa_system/src/model/clinical_encoders.py

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from transformers import AutoModel

class MedicalImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b4', in_channels=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.base.extract_features(x)
        x = self.adaptive_pool(x)  # Reduce spatial dimensions
        return x.flatten(1)


class ClinicalTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # ✅ Use ClinicalBERT (from Emily Alsentzer)
        self.biobert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.projection = nn.Linear(768, 512)  # Match expected fusion dim

    def forward(self, input_ids, attention_mask=None):
        outputs = self.biobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_token = outputs.last_hidden_state[:, 0]  # [CLS] token representation
        return self.projection(cls_token)
