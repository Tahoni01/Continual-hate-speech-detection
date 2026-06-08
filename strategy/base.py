# strategy/base.py


class BaseStrategy:
    # all strategies plug into the trainer via these two hooks:
    # compute_loss() is called every step, on_task_end() is called at drift detection

    @property
    def name(self):
        return type(self).__name__.replace("Strategy", "")

    def on_task_end(self, trainer, stream):
        pass

    def compute_loss(self, trainer, inputs, labels, logits):
        return 0.0
