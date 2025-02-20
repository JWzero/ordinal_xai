from interpretability.base_interpretation import BaseInterpretation

class DummyInterpretation(BaseInterpretation):
    """Dummy interpretation method that does nothing."""

    def explain_instance(self, model, instance):
        """Returns a placeholder explanation."""
        return "No explanation provided."