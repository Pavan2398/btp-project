# ecg_vqa_system/src/utils/medical_profiler.py
import time
import torch

class MedicalProfiler:
    def __init__(self):
        self.events = {}
        
    def record(self, name):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.events[name] = time.time()
        
    def elapsed(self, start, end):
        return self.events[end] - self.events[start]