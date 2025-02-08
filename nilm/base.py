import torch
import numpy as np
from abc import ABC, abstractmethod
import torch.nn as nn
import torch.optim as optim
from ..metrics import f1_score, teca_score, modified_f1_score, jaccard_score, modified_jaccard_score


class BaseModel(ABC, nn.Module):

    def __init__(self):
        super().__init__()
        self.best_mf_score = -float('inf')
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_scores": {
                "F1": [],
                "MF": [],
                "J": [],
                "MJ": [],
                "TECA": []
            },
            "val_scores": {
                "F1": [],
                "MF": [],
                "J": [],
                "MJ": [],
                "TECA": []
            },
        }

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def fit(
        self,
        train_loader,
        val_loader=None,
        optimizer=optim.Adam,
        n_epochs=100,
        lr=1e-3,
        weight_decay=0,
        scheduler=None,
        device=None,
        checkpoint_path_last=None,
        checkpoint_path_MJ=None,
        n_gpus=1,
        **kwargs,
    ):
        # Device setup
        device = torch.device(device if device else 'cuda' if torch.cuda.
                              is_available() else 'cpu')
        if n_gpus > 1 and torch.cuda.device_count() >= n_gpus:
            self = torch.nn.DataParallel(self, device_ids=list(range(n_gpus)))

        self.to(device)

        optim_kwargs = {
            k[len('optimizer__'):]: v
            for k, v in kwargs.items() if k.startswith('optimizer__')
        }
        scheduler_kwargs = {
            k[len('scheduler__'):]: v
            for k, v in kwargs.items() if k.startswith('scheduler__')
        }

        optimizer = optimizer(self.parameters(),
                              lr=lr,
                              weight_decay=weight_decay,
                              **optim_kwargs)
        if scheduler is not None:
            scheduler = scheduler(optimizer, **scheduler_kwargs)

        if isinstance(self, nn.DataParallel):
            module = self.module
        else:
            module = self

        # Training loop
        for epoch in range(n_epochs):
            self.train()
            running_loss = 0.0
            all_train_preds, all_train_labels = [], []

            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()

                outputs = self(inputs)
                loss = module.loss(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                all_train_preds.append(outputs.detach().cpu().numpy())
                all_train_labels.append(labels.cpu().numpy())

            # Calculate and store training loss and scores
            epoch_train_loss = running_loss / len(train_loader)
            module.history["train_loss"].append(epoch_train_loss)
            Y_train_pred = np.vstack(all_train_preds)
            Y_train_true = np.vstack(all_train_labels)
            train_scores = module.scores(Y_train_true, Y_train_pred)

            for metric, value in train_scores.items():
                module.history["train_scores"][metric].append(value)

            # Validation step if applicable
            if val_loader:
                val_loss, val_scores = module._validate(val_loader, device)
                module.history["val_loss"].append(val_loss)
                for metric, value in val_scores.items():
                    module.history["val_scores"][metric].append(value)

                val_MJ = val_scores['MJ']
                print(
                    f"Epoch {epoch+1}/{n_epochs}, Train Loss: {epoch_train_loss:.4f}, Validation Loss: {val_loss:.4f}, J: {val_scores['J']:.4f}, MJ: {val_MJ:.4f}, TECA: {val_scores['TECA']:.4f}"
                )

                # Save checkpoint for the highest validation MJ
                if val_MJ > module.best_mf_score:
                    module.best_mf_score = val_MJ
                    if checkpoint_path_MJ:
                        module._save_checkpoint(checkpoint_path_MJ, optimizer,
                                                scheduler)
                        print(
                            f"Checkpoint (highest MJ) saved at epoch {epoch+1} with MJ: {module.best_mf_score:.4f}"
                        )

            else:
                print(
                    f"Epoch {epoch+1}/{n_epochs}, Train Loss: {epoch_train_loss:.4f}"
                )

            # Scheduler step
            if scheduler:
                scheduler.step(epoch_train_loss)

            if checkpoint_path_last:
                module._save_checkpoint(checkpoint_path_last, optimizer,
                                        scheduler)
                print(f"Last checkpoint saved.")

    def _validate(self, data_loader, device):
        """Validation step within fit."""
        self.eval()
        all_val_preds, all_val_labels = [], []
        running_val_loss = 0.0

        if isinstance(self, nn.DataParallel):
            module = self.module
        else:
            module = self

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                loss = module.loss(outputs, labels)

                running_val_loss += loss.item()

                all_val_preds.append(outputs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

        val_loss = running_val_loss / len(data_loader)
        Y_val_pred = np.vstack(all_val_preds)
        Y_val_true = np.vstack(all_val_labels)
        val_scores = module.scores(Y_val_true, Y_val_pred)

        return val_loss, val_scores

    def normalize(self, x):
        return x

    def postprocess(self, y):
        return y

    def predict(self, test_loader, device=None):
        """Generate predictions for input data X."""
        device = torch.device(device if device else 'cuda' if torch.cuda.
                              is_available() else 'cpu')
        self.to(device)

        Y_pred = []
        self.eval()

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = self(inputs)
                P = targets.numpy().sum(1, keepdims=True)

                outputs = outputs.cpu().numpy()
                outputs = np.where(P * outputs > self.thresh, outputs, 0)
                outputs = outputs / (outputs.sum(1, keepdims=True) + 1e-9)
                outputs = P * outputs

                Y_pred.append(outputs)

        Y_pred = np.vstack(Y_pred)

        return Y_pred

    def _save_checkpoint(self, checkpoint_path, optimizer, scheduler):
        """Save model, optimizer, scheduler states, and training history for reproducibility."""
        checkpoint = {
            'model_state_dict':
            self.module.state_dict()
            if isinstance(self, nn.DataParallel) else self.state_dict(),
            'optimizer_state_dict':
            optimizer.state_dict(),
            'scheduler_state_dict':
            scheduler.state_dict() if scheduler else None,
            'history':
            self.history,
            'best_mf_score':
            self.best_mf_score,
        }
        torch.save(checkpoint, checkpoint_path)

    def loss(self, outputs, labels):
        labels = labels / labels.sum(1, keepdims=True)
        loss_fn = nn.BCELoss()

        return loss_fn(outputs, labels)

    def scores(self, Y_true, Y_pred):
        P = Y_true.sum(1, keepdims=True)
        Y_pred = np.where(P * Y_pred > self.thresh, Y_pred, 0)
        Y_pred = Y_pred / Y_pred.sum(1, keepdims=True)
        Y_pred = P * Y_pred

        F1 = f1_score(Y_true, Y_pred)
        MF = modified_f1_score(Y_true, Y_pred)
        TECA = teca_score(Y_true, Y_pred)
        J = jaccard_score(Y_true, Y_pred)
        MJ = modified_jaccard_score(Y_true, Y_pred)
        scores = dict(F1=F1, MF=MF, J=J, MJ=MJ, TECA=TECA)

        return scores
