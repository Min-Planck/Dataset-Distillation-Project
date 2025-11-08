from abc import ABC, abstractmethod

# Distill into Model
class IDatasetDistillation(ABC): 

    @abstractmethod
    def distillation(self): 
        pass

    @abstractmethod
    def evaluate(self, model, ipc: int) -> float:
        pass

    @abstractmethod
    def generate_sample(self, ipc: int, save_root: str):
        pass


# Distill into Image
class IDatasetCondensation(ABC):
    @abstractmethod
    def condensation(self, distillation_steps: int, network_step: int):
        pass

    @abstractmethod
    def evaluate(self, num_train_epochs: int) -> float:
        pass