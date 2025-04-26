# ecg_vqa_system/src/data/clinical_transforms.py
import torchvision.transforms as T

class ClinicalTransforms:
    def __init__(self, img_size=512):
        self.train = T.Compose([
            T.RandomRotation(5),
            T.RandomAdjustSharpness(2),
            T.RandomAutocontrast(),
            T.Lambda(lambda x: x.convert('L')),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        
        self.val = T.Compose([
            T.Lambda(lambda x: x.convert('L')),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])