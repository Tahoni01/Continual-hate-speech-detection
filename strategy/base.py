# strategy/base.py
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """Interfaccia comune per tutte le strategie anti-forgetting."""

    def on_task_end(self, trainer, stream, label_map):
        """
        Hook chiamato dal trainer al termine di ogni task.
        Override nelle strategie che richiedono aggiornamenti post-task
        (es. EWC: ricalcola Fisher, aggiorna params di riferimento).
        """
        pass

    def compute_loss(self, trainer, inputs, labels, logits):
        """
        Contributo aggiuntivo alla loss per questa strategia.
        Ritorna 0.0 se la strategia non modifica la loss direttamente.
        """
        return 0.0