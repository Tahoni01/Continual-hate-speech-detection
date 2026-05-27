import torch
from torch.optim import AdamW
from tqdm import tqdm


class ContinualTrainer:
    def __init__(self, model, tokenizer, device="cuda", lr=2e-5):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.replay = None
        self.ewc = None
        self.distillation = None

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
    # LOSS (base + EWC + distillation)
    # -----------------------
    def _compute_loss(self, logits, labels, inputs=None):
        loss = self.ce_loss(logits, labels)

        if self.ewc is not None:
            loss = loss + self.ewc.ewc_loss()

        if self.distillation is not None and inputs is not None:
            with torch.no_grad():
                teacher_logits = self.distillation.old_model(**inputs).logits
            T     = self.distillation.temperature
            alpha = self.distillation.alpha
            distill_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(logits / T, dim=-1),
                torch.nn.functional.softmax(teacher_logits / T, dim=-1),
                reduction="batchmean"
            ) * (T ** 2)
            loss = alpha * loss + (1 - alpha) * distill_loss

        return loss

    # -----------------------
    # SINGLE TRAIN STEP
    # -----------------------
    def train_step(self, batch, label_map):
        self.model.train()

        inputs, labels = self.prepare_batch(batch, label_map)
        outputs = self.model(**inputs)
        loss = self._compute_loss(outputs.logits, labels, inputs)

        # REPLAY: allena sui campioni del buffer + aggiorna il buffer
        if self.replay is not None:
            replay_inputs, replay_labels = self.replay.sample(len(labels))
            if replay_inputs is not None:
                replay_inputs  = {k: v.to(self.device) for k, v in replay_inputs.items()}
                replay_labels  = replay_labels.to(self.device)
                replay_outputs = self.model(**replay_inputs)
                loss = loss + self._compute_loss(replay_outputs.logits, replay_labels)
            self.replay.update_buffer(inputs, labels)

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
                print(f"[batch {i+1}] avg loss: {avg:.4f}")

        # restituisce tensori flat direttamente
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