# ecg_vqa_system/src/data/clinical_loader.py
import os
import json
from functools import lru_cache
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np

class ClinicalECGDataset(Dataset):
    def __init__(self, json_path, image_dir, tokenizer_name, max_length=128):
        self.data = self._load_medical_json(json_path)
        self.image_dir = image_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def _load_medical_json(self, path):
        with open(path) as f:
            return [self._process_item(qa) for qa in json.load(f)]

    def _process_item(self, item):
        return {
            'ecg_path': item['ecg_path'][0],
            'question': item['question'],
            'answer': 1 if item['answer'][0].lower() == 'yes' else 0
        }

    @lru_cache(maxsize=1000)
    def _load_ecg_image(self, path):
        if not path.endswith('.png'):
            path += '.png'
        image = Image.open(os.path.join(self.image_dir, path)).convert('L')
        return self._medical_transform(image)


    def _medical_transform(self, img):
        img_array = np.array(img)  # convert PIL image to numpy array
        return torch.stack([
            torch.tensor(img_array).float().div(255).sub(0.5).div(0.5)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'image': self._load_ecg_image(item['ecg_path']),
            'input_ids': self.tokenizer(
                item['question'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )['input_ids'].squeeze(),
            'answer': torch.tensor(item['answer'], dtype=torch.long)
        }