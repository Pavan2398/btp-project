# ecg_vqa_system/src/model/clinical_vqa.py
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .clinical_encoders import MedicalImageEncoder, ClinicalTextEncoder
from .multimodal_fusion import MedicalCrossAttention, DiagnosticGate
from .clinical_heads import ClinicalClassifier

class ClinicalVQAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = MedicalImageEncoder()
        self.text_encoder = ClinicalTextEncoder()
        self.cross_attn = MedicalCrossAttention()
        self.diagnostic_gate = DiagnosticGate(input_dim=512)
        self.classifier = ClinicalClassifier()

    def forward(self, images, input_ids):
      
        img_features = cp.checkpoint(self.image_encoder, images)
        txt_features = cp.checkpoint(self.text_encoder, input_ids)

       
        fused_features = self.cross_attn(img_features, txt_features)

   
        gated_features = self.diagnostic_gate(fused_features)

       
        return self.classifier(gated_features)
