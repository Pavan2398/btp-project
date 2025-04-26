# ecg_vqa_system/src/configs/clinical_config.py
class ClinicalConfig:
    def __init__(self):
        self.img_size = 512
        self.batch_size = 32
        self.lr = 3e-5
        self.epochs = 20
        self.tokenizer = "monologg/biobert_v1.1_pubmed"
        self.model_save_path = "clinical_models/saved_models/best_model.pt"