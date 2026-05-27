# strategy/distillation.py
import torch
import torch.nn.functional as F
from strategy.base import BaseStrategy

class DistillationStrategy(BaseStrategy):
    def __init__(self, temperature=2.0, alpha=0.5):
        self.old_model   = None   # snapshot del modello precedente
        self.temperature = temperature
        self.alpha       = alpha

    def on_task_end(self, trainer, stream, label_map):
        """
        Salva uno snapshot del modello corrente come teacher
        per il task successivo.
        """
        import copy
        self.old_model = copy.deepcopy(trainer.model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False
        print("[Distillation] Snapshot teacher aggiornato.")

    def compute_loss(self, trainer, inputs, labels, logits):
        if self.old_model is None:
            return 0.0  # primo task, nessun teacher disponibile

        self.old_model = self.old_model.to(trainer.device)

        with torch.no_grad():
            teacher_logits = self.old_model(**inputs).logits

        T = self.temperature
        distill_loss = F.kl_div(
            F.log_softmax(logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean"
        ) * (T ** 2)

        # nota: la CE loss base viene già calcolata nel trainer
        # qui restituiamo solo il contributo distillation scalato
        return (1 - self.alpha) * distill_loss