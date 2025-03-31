from abc import abstractmethod
import torch

class SpecialTransform():
    @abstractmethod
    def direct(data):
        pass
    
    @abstractmethod
    def reverse(data):
        pass
    
class LogTransform(SpecialTransform):
    def direct(data):
        return torch.log(1.0 + data)
    def reverse(data):
        return torch.exp(data) - 1.0