class BaseStrategy:
    # two hooks called by the trainer:
    # on_task_end() at drift with pre-split old/new batches
    # compute_loss() at every training step

    @property
    def name(self):
        return type(self).__name__.replace("Strategy", "")

    def on_task_end(self, trainer, old_batches, new_batches):
        pass

    def compute_loss(self, trainer, inputs, labels, logits):
        return 0.0

    def old_task_batches(self, trainer, batch_size=32):
        """Pre-encoded (enc, labels) batches for tuner forgetting eval.
        Returns None → tuner falls back to old_batches DataFrames."""
        return None