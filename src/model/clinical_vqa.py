# ecg_vqa_system/src/model/clinical_vqa.py
import torch
import torch.nn as nn

class ClinicalVQAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = MedicalImageEncoder()
        self.text_encoder = ClinicalTextEncoder()
        self.cross_attn = MedicalCrossAttention()
        self.diagnostic_gate = DiagnosticGate()
        self.classifier = ClinicalClassifier()

    def forward(self, images, input_ids):
        # Image processing
        img_features = self.image_encoder(images)
        
        # Text processing
        txt_features = self.text_encoder(input_ids)
        
        # Multimodal fusion
        fused_features = self.cross_attn(img_features, txt_features)
        
        # Diagnostic gating
        gated_features = self.diagnostic_gate(fused_features)
        
        # Classification
        return self.classifier(gated_features)