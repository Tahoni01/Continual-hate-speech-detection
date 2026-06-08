import random
import torch
import torch.nn.functional as F
from strategy.base import BaseStrategy


class ReplayStrategy(BaseStrategy):
    def __init__(self, buffer_size=500, n_classes=3):
        self.buffer_size = buffer_size
        self.n_classes   = n_classes
        self.buffers     = {c: [] for c in range(n_classes)}
        self.seen        = {c: 0 for c in range(n_classes)}
        self.active      = False

    @property
    def slot_size(self):
        return self.buffer_size // self.n_classes

    @property
    def n_stored(self):
        return sum(len(b) for b in self.buffers.values())

    def on_task_end(self, trainer, old_batches, new_batches):
        self.active = True

    def _reservoir_update(self, c, sample):
        # Uniform reservoir sampling, each sample has equal probability slot_size/seen.
        buf = self.buffers[c]
        self.seen[c] += 1
        if len(buf) < self.slot_size:
            buf.append(sample)
        else:
            j = random.randint(0, self.seen[c] - 1)
            if j < self.slot_size:
                buf[j] = sample

    def _pad_and_stack(self, ids_list, mask_list):
        max_len = max(t.shape[0] for t in ids_list)
        pad = lambda ts, val: torch.stack(
            [F.pad(t, (0, max_len - t.shape[0]), value=val) for t in ts]
        )
        return {"input_ids": pad(ids_list, 1), "attention_mask": pad(mask_list, 0)}

    def _update(self, inputs, labels):
        ids  = inputs["input_ids"].detach().cpu()
        mask = inputs["attention_mask"].detach().cpu()
        lbls = labels.detach().cpu()
        for i in range(len(lbls)):
            self._reservoir_update(lbls[i].item(), (ids[i], mask[i], lbls[i]))

    def _sample(self, n_per_class):
        ids_list, mask_list, lbl_list = [], [], []
        for buf in self.buffers.values():
            if not buf:
                continue
            for ids, mask, lbl in random.sample(buf, min(n_per_class, len(buf))):
                ids_list.append(ids)
                mask_list.append(mask)
                lbl_list.append(lbl)
        if not ids_list:
            return None, None
        return self._pad_and_stack(ids_list, mask_list), torch.stack(lbl_list)

    def old_task_batches(self, trainer, batch_size=32):
        samples = [(ids, mask, lbl)
                   for buf in self.buffers.values()
                   for ids, mask, lbl in buf]
        if not samples:
            return None
        batches = []
        for i in range(0, len(samples), batch_size):
            chunk = samples[i:i + batch_size]
            ids_list, mask_list, lbl_list = zip(*chunk)
            batches.append((
                self._pad_and_stack(ids_list, mask_list),
                torch.stack(lbl_list),
            ))
        return batches

    def compute_loss(self, trainer, inputs, labels, logits):
        self._update(inputs, labels)
        if not self.active:
            return 0.0

        n_per_class = max(1, len(labels) // self.n_classes)
        replay_inputs, replay_labels = self._sample(n_per_class)
        if replay_inputs is None:
            return 0.0

        replay_inputs = {k: v.to(trainer.device) for k, v in replay_inputs.items()}
        return trainer.model(**replay_inputs, labels=replay_labels.to(trainer.device)).loss