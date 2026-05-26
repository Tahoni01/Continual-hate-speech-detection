import torch
from torch.optim import AdamW
from tqdm import tqdm

class ContinualTrainer:
    def __init__(self, model, tokenizer, device='cuda',
                 lr=2e-5, strategy=None, label_map=None):

        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer

        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.strategy = strategy

        self.label_map = label_map or {
            "hatespeech": 0,
            "offensive": 1,
            "normal": 2
        }

        self.loss_fn = torch.nn.CrossEntropyLoss()

    # -----------------------
    def prepare_batch(self, batch):
        encodings = self.tokenizer(
            batch["text"].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        labels = torch.tensor(
            [self.label_map[x] for x in batch["label"].tolist()],
            dtype=torch.long
        ).to(self.device)

        return encodings, labels

    # -----------------------
    def train_step(self, batch):
        self.model.train()
        inputs, labels = self.prepare_batch(batch)

        # forward
        outputs = self.model(**inputs)

        loss = self.loss_fn(outputs.logits, labels)

        # =========================
        # STRATEGIES COMBINATE
        # =========================

        # EWC
        if self.strategy and hasattr(self.strategy, "ewc_loss"):
            loss = loss + self.strategy.ewc_loss()

        # DISTILLATION (richiede model + inputs)
        if self.strategy and hasattr(self.strategy, "distill"):
            loss = self.strategy.distill(
                self.model,
                inputs,
                labels,
                self.loss_fn,
                self.device
            )

        # BACKPROP
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # REPLAY BUFFER UPDATE
        if self.strategy and hasattr(self.strategy, "update_buffer"):
            self.strategy.update_buffer(inputs, labels)

        preds = torch.argmax(outputs.logits, dim=1)

        return loss.item(), preds.cpu(), labels.cpu()

    # -----------------------
    def train_continual(self, stream, log_every=10):

        losses = []

        for i, batch in enumerate(tqdm(stream)):

            loss, preds, labels = self.train_step(batch)
            losses.append(loss)

            if (i + 1) % log_every == 0:
                avg = sum(losses[-log_every:]) / log_every
                print(f"[{i+1}] loss: {avg:.4f}")

        return losses

    # -----------------------
    def evaluate(self, stream):

        self.model.eval()

        preds_all, labels_all = [], []

        with torch.no_grad():
            for batch in stream:

                inputs, labels = self.prepare_batch(batch)

                outputs = self.model(**inputs)

                preds = torch.argmax(outputs.logits, dim=1)

                preds_all.append(preds.cpu())
                labels_all.append(labels.cpu())

        return preds_all, labels_all