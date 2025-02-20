from abc import ABC, abstractmethod

class BaseInterpretation(ABC):
    """Abstract base class for interpretability methods."""

    @abstractmethod
    def explain_instance(self, model, instance):
        """Explain a single instance. Must be implemented by subclasses."""
        pass