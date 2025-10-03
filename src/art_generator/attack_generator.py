from abc import ABC, abstractmethod
import numpy as np
 
class AttackGenerator(ABC):

    def __init__(self):
        self.attack = None
        self.attack_params = None

        
    @abstractmethod
    def generate(self, classifier, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")