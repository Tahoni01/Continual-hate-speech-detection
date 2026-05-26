import torch
from torch.optim import AdamW
from tqdm import tqdm

class ContinualTrainer:
    def __init__(self, model, tokenizer, device="cuda", lr=2e-5):

        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer

        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # OPTIONAL COMPONENTS (set externally)
        self.replay = None
        self.ewc = None
        self.distillation = None

    # --------------------------
    # batch encoding
    # --------------------------
    def prepare_batch(self, batch, label_map):

        encodings = self.tokenizer(
            batch["text"].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        labels = torch.tensor(
            [label_map[x] for x in batch["label"].tolist()],
            dtype=torch.long
        ).to(self.device)

        return encodings, labels

    # --------------------------
    # TRAIN STEP
    # --------------------------
    def train_step(self, batch, label_map):

        self.model.train()

        inputs, labels = self.prepare_batch(batch, label_map)

        outputs = self.model(**inputs)
        logits = outputs.logits

        # base loss
        loss = self.loss_fn(logits, labels)

        # --------------------------
        # EWC
        # --------------------------
        if self.ewc is not None:
            loss = loss + self.ewc.ewc_loss()

        # --------------------------
        # DISTILLATION
        # --------------------------
        if self.distillation is not None:

            with torch.no_grad():
                old_outputs = self.distillation["old_model"](**inputs)

            T = self.distillation.get("temperature", 2.0)
            alpha = self.distillation.get("alpha", 0.5)

            soft_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(logits / T, dim=-1),
                torch.nn.functional.softmax(old_outputs.logits / T, dim=-1),
                reduction="batchmean"
            ) * (T * T)

            loss = alpha * loss + (1 - alpha) * soft_loss

        # --------------------------
        # BACKPROP
        # --------------------------
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --------------------------
        # REPLAY UPDATE
        # --------------------------
        if self.replay is not None:
            self.replay.update_buffer(inputs, labels)

        preds = torch.argmax(logits, dim=1)

        return loss.item(), preds.cpu(), labels.cpu()

    # --------------------------
    # TRAIN LOOP
    # --------------------------
    def train_continual(self, stream, label_map, log_every=10):

        losses = []
        preds_list = []
        labels_list = []

        for i, batch in enumerate(tqdm(stream)):

            loss, preds, labels = self.train_step(batch, label_map)

            losses.append(loss)
            preds_list.append(preds)
            labels_list.append(labels)

            if (i + 1) % log_every == 0:
                print(f"[{i+1}] loss: {sum(losses[-log_every:]) / log_every:.4f}")

        return losses, preds_list, labels_list

    # --------------------------
    # EVALUATION
    # --------------------------
    def evaluate(self, stream, label_map):

        self.model.eval()

        preds_list, labels_list = [], []

        with torch.no_grad():

            for batch in stream:

                inputs, labels = self.prepare_batch(batch, label_map)

                outputs = self.model(**inputs)

                preds = torch.argmax(outputs.logits, dim=1)

                preds_list.append(preds.cpu())
                labels_list.append(labels.cpu())

        return preds_list, labels_list