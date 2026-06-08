import torch
from collections import deque
from torch.optim import AdamW
from tqdm import tqdm
from river.drift import ADWIN


class ContinualTrainer:
    def __init__(self, model, tokenizer, device, lr=2e-5, lr_head=1e-4,
                 adwin_delta=0.002, buffer_size=200, lr_decay=0.1):
        self.model     = model.to(device)
        self.tokenizer = tokenizer
        self.device    = device
        self.lr        = lr
        self.lr_head   = lr_head
        self.lr_decay  = lr_decay
        self.optimizer = self._make_optimizer()
        self.scaler    = torch.amp.GradScaler()

        self.strategies = []
        self.eval_log   = []
        self.tuner      = None

        self.detector       = ADWIN(delta=adwin_delta)
        self.adwin_delta    = adwin_delta
        self.drift_detected = False
        self.drift_batch    = None

        # rolling buffer of recent batches — passed to on_task_end and tuner at drift
        self._recent_batches = deque(maxlen=buffer_size)

    def _split_recent_batches(self):
        batches = list(self._recent_batches)
        if not batches:
            return [], []
        sources = [b["source"].iloc[0] for b in batches]
        if len(set(sources)) == 1:
            return batches, []
        split = next(i for i in range(1, len(sources)) if sources[i] != sources[i-1])
        return batches[:split], batches[split:]

    def _make_optimizer(self):
        if hasattr(self.model, "parameter_groups"):
            return AdamW(self.model.parameter_groups(self.lr, self.lr_head))
        return AdamW(self.model.parameters(), lr=self.lr)

    # ── strategy management ──────────────────────────────────────────────────

    def add_strategy(self, strategy):
        self.strategies.append(strategy)
        return self

    def set_tuner(self, tuner):
        self.tuner = tuner
        return self

    def on_task_end(self, old_batches=None, new_batches=None):
        if old_batches is None:
            old_batches, new_batches = self._split_recent_batches()
        for s in self.strategies:
            s.on_task_end(self, old_batches, new_batches or [])

    # ── checkpoint ──────────────────────────────────────────────────────────

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device,
                                              weights_only=True))
        self.optimizer      = self._make_optimizer()
        self.scaler         = torch.amp.GradScaler()
        self.strategies     = []
        self.eval_log       = []
        self.tuner          = None
        self.detector       = ADWIN(delta=self.adwin_delta)
        self.drift_detected = False
        self.drift_batch    = None
        self._recent_batches.clear()

    # ── encoding ─────────────────────────────────────────────────────────────

    def _encode(self, batch, label_map):
        enc = self.tokenizer(
            batch["text"].tolist(),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        labels = torch.tensor(
            [label_map[l] for l in batch["label"].tolist()],
            dtype=torch.long,
            device=self.device,
        )
        return enc, labels

    # ── evaluation ───────────────────────────────────────────────────────────

    def _run_eval(self, val_stream, label_map, return_preds=False):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_stream:
                enc, labels = self._encode(batch, label_map)
                preds = self.model(**enc).logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        self.model.train()

        p, l = torch.cat(all_preds), torch.cat(all_labels)
        return (p, l) if return_preds else (p == l).float().mean().item()

    def evaluate(self, val_stream, label_map):
        return self._run_eval(val_stream, label_map, return_preds=True)

    # ── drift detection ──────────────────────────────────────────────────────

    def _update_detector(self, preds, labels, batch_idx):
        error_rate = (preds != labels).float().mean().item()
        self.detector.update(error_rate)

        if self.detector.drift_detected and not self.drift_detected:
            self.drift_detected = True
            self.drift_batch    = batch_idx
            print(f"\n[ADWIN] Drift detected at batch {batch_idx}")

            # decay LR so the model doesn't overwrite old knowledge too aggressively
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["lr"] * self.lr_decay
            print(f"[ADWIN] LR decayed by {self.lr_decay}x")

            old_batches, new_batches = self._split_recent_batches()
            self.on_task_end(old_batches, new_batches)
            if self.tuner is not None:
                self.tuner.tune(self, old_batches, new_batches)

    # ── training loop ────────────────────────────────────────────────────────

    def train(self, stream, label_map, val_streams: dict, eval_every=50):
        """
        Returns a dict with 'task' (CE on current batch only) and 'total'
        (CE + strategy contributions) loss curves for plotting.
        """
        self._last_label_map = label_map
        self.model.train()
        task_losses, total_losses = [], []

        for i, batch in enumerate(tqdm(stream, desc="Training")):
            enc, labels = self._encode(batch, label_map)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out        = self.model(**enc, labels=labels)
                total_loss = out.loss + sum(
                    s.compute_loss(self, enc, labels, out.logits)
                    for s in self.strategies
                )

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            task_losses.append(out.loss.item())
            total_losses.append(total_loss.item())

            with torch.no_grad():
                preds = out.logits.detach().argmax(dim=1)
            self._update_detector(preds, labels, i)

            self._recent_batches.append(batch)

            if i % eval_every == 0:
                entry = {"batch": i}
                for name, vs in val_streams.items():
                    entry[name] = self._run_eval(vs, label_map)
                self.eval_log.append(entry)

        entry = {"batch": len(stream)}
        for name, vs in val_streams.items():
            entry[name] = self._run_eval(vs, label_map)
        self.eval_log.append(entry)

        return {"task": task_losses, "total": total_losses}