import torch
from torch.optim import AdamW
from tqdm import tqdm
from strategy.replay import ReplayStrategy


class ContinualTrainer:
    def __init__(self, model, tokenizer, device="cuda", lr=2e-5):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.strategies = []  # ← fix: era mancante

    # -----------------------
    # STRATEGY
    # -----------------------
    def add_strategy(self, strategy):
        self.strategies.append(strategy)

    def on_task_end(self, stream, label_map):
        for strategy in self.strategies:
            strategy.on_task_end(self, stream, label_map)

    # -----------------------
    # ENCODE BATCH
    # -----------------------
    def prepare_batch(self, batch, label_map):
        enc = self.tokenizer(
            batch["text"].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        labels = torch.tensor(
            [label_map[x] for x in batch["label"].tolist()],
            dtype=torch.long,
            device=self.device
        )
        return enc, labels

    # -----------------------
    # LOSS (base + strategie)
    # -----------------------
    def _compute_loss(self, inputs, labels, logits):
        loss = self.ce_loss(logits, labels)
        for strategy in self.strategies:
            loss = loss + strategy.compute_loss(self, inputs, labels, logits)
        return loss

    # -----------------------
    # SINGLE TRAIN STEP
    # -----------------------
    def train_step(self, batch, label_map):
        self.model.train()

        inputs, labels = self.prepare_batch(batch, label_map)
        outputs = self.model(**inputs)
        loss = self._compute_loss(inputs, labels, outputs.logits)  # ← fix: ordine args

        # REPLAY — gestito tramite strategia
        for strategy in self.strategies:
            if isinstance(strategy, ReplayStrategy):
                replay_inputs, replay_labels = strategy.sample(len(labels))
                if replay_inputs is not None:
                    replay_inputs  = {k: v.to(self.device) for k, v in replay_inputs.items()}
                    replay_labels  = replay_labels.to(self.device)
                    replay_outputs = self.model(**replay_inputs)
                    loss = loss + self.ce_loss(replay_outputs.logits, replay_labels)
                strategy.update_buffer(inputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        preds = torch.argmax(outputs.logits, dim=1)
        return loss.item(), preds.detach().cpu(), labels.detach().cpu()

    # -----------------------
    # TRAIN LOOP
    # -----------------------
    def train_continual(self, stream, label_map, log_every=10):
        losses, preds_list, labels_list = [], [], []

        for i, batch in enumerate(tqdm(stream)):
            loss, preds, labels = self.train_step(batch, label_map)
            losses.append(loss)
            preds_list.append(preds)
            labels_list.append(labels)

            if (i + 1) % log_every == 0:
                avg = sum(losses[-log_every:]) / log_every
                #print(f"[batch {i+1}] avg loss: {avg:.4f}")

        return losses, torch.cat(preds_list), torch.cat(labels_list)

    # -----------------------
    # EVAL
    # -----------------------
    def evaluate(self, stream, label_map):
        self.model.eval()
        preds_list, labels_list = [], []

        with torch.no_grad():
            for batch in stream:
                inputs, labels = self.prepare_batch(batch, label_map)
                preds = torch.argmax(self.model(**inputs).logits, dim=1)
                preds_list.append(preds.cpu())
                labels_list.append(labels.cpu())

        return torch.cat(preds_list), torch.cat(labels_list)