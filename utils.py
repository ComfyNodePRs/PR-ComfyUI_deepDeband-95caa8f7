import gc
import torch

def clearCache(self):
    mm.soft_empty_cache()
    if self.transformer:
        self.transformer.cpu()
        del self.transformer
    if self.tokenizer:
        del self.tokenizer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    self.tokenizer = None
    self.transformer = None
    self.model_name = None
    self.precision = None
    self.quantization = None