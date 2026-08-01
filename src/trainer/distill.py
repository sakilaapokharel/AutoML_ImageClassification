import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.automl.utils import print_header, print_step
from src.dataset.loaders import load_datasets
from src.dataset.transforms import build_transform
from src.models.embedders import Embedder
from src.models.encoders import get_encoder
from src.models.reducers import PCAReducer
from src.models.student import Student
from src.models.tabpfn import TabPFNModel

class Distiller:

    def __init__(
        self,
        batch_size=32,
        epochs=20,
        lr=1e-3,
        alpha=0.7,
        temperature=8,
        patience=8,
        val_ratio=0.2,
        device="cuda",
        checkpoint_root="checkpoints",
        train_fidelity=-1,
        seed=42,
        warm_epochs=50,
        tabpfn_model=TabPFNModel,
        embedder=Embedder,
        dataset_name=None,
        encoder_name=None,
        embedding_dim=None,
        ditill_flag=False,
    ):

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.warm_epochs = warm_epochs

        self.alpha = alpha
        self.temperature = temperature

        self.patience = patience
        self.val_ratio = val_ratio

        self.device = device

        self.checkpoint_root = Path(checkpoint_root)
        self.train_fidelity = train_fidelity

        self.seed = seed
        self.tabpfn_model = tabpfn_model
        self.embedder = embedder
        self.dataset_name = dataset_name
        self.encoder_name = encoder_name
        self.embedding_dim = embedding_dim
        self.ditill_flag = ditill_flag

    # ------------------------------------------------
    # Teacher
    # ------------------------------------------------

    def _train_teacher(
        self,
        X_train,
        y_train,
        config,
        seed,
    ):

        reducer = None

        if config["embedding_dim"] is not None:

            print_step("Applying PCA")

            reducer = PCAReducer(
                config["embedding_dim"],
                seed,
            )

            X_train = reducer.fit_transform(X_train)

        print_step("Training TabPFN teacher")

        teacher = self.tabpfn_model

        teacher.fit(
            X_train,
            y_train,
        )

        return (
            teacher,
            reducer,
        )

    def _teacher_probs(
        self,
        teacher,
        reducer,
        X,
    ):
        if reducer != None:
            X = reducer.transform(X)

        # TabPFNModel.predict_proba already batches internally (see
        # models/tabpfn.py) -- delegate to it instead of chunking again here.
        return teacher.predict_proba(
            X,
        )

    def _teacher_predict(
        self,
        teacher,
        reducer,
        X,
    ):
        if reducer != None:
            X = reducer.transform(X)

        return teacher.predict(
            X,
        )

    # ------------------------------------------------
    # Student
    # ------------------------------------------------

    def _train_student(
        self,
        train_dataset,
        val_dataset,
        student,
    ):

        print_step("Training CNN student")

        #################################################
        # Split dataset
        #################################################

        indices = list(range(len(train_dataset)))

        train_idx, val_idx = train_test_split(
            indices,
            test_size=self.val_ratio,
            random_state=self.seed,
        )

        train_ds = Subset(
            train_dataset,
            train_idx,
        )

        val_ds = Subset(
            val_dataset,
            val_idx,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

        #################################################
        # Model
        #################################################

        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

        ce = nn.CrossEntropyLoss()

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        #################################################
        # Training
        #################################################

        for epoch in range(self.epochs):

            student.train()

            train_loss = 0.0
            correct = 0
            total = 0

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()

                logits = student(x)

                loss = ce(
                    logits,
                    y,
                )

                loss.backward()

                optimizer.step()

                train_loss += loss.item()

                pred = logits.argmax(dim=1)

                correct += (pred == y).sum().item()
                total += y.size(0)

            train_loss /= len(train_loader)
            train_acc = correct / total

            #################################################
            # Validation
            #################################################

            student.eval()

            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():

                for x, y in val_loader:

                    x = x.to(self.device)

                    y = y.to(self.device)

                    logits = student(x)

                    loss = ce(
                        logits,
                        y,
                    )

                    val_loss += loss.item()

                    pred = logits.argmax(dim=1)

                    correct += (pred == y).sum().item()
                    total += y.size(0)

            val_loss /= len(val_loader)
            val_acc = correct / total

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch+1:03d}/{self.epochs} | "
                f"TrainLoss={train_loss:.4f} | "
                f"TrainAcc={100*train_acc:.2f}% | "
                f"ValLoss={val_loss:.4f} | "
                f"ValAcc={100*val_acc:.2f}%"
            )

            #################################################
            # Early stopping
            #################################################

            if val_loss < best_loss:

                best_loss = val_loss

                best_state = {
                    k: v.cpu().clone() for k, v in student.state_dict().items()
                }

                patience_counter = 0

            else:

                patience_counter += 1

            if patience_counter >= self.patience:

                print_step("Early stopping")

                break

        #################################################
        # Restore best model
        #################################################

        if best_state is not None:

            student.load_state_dict(best_state)

        history_dir = Path("checkpoints") / self.dataset_name

        history_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez(
            history_dir / "training_history.npz",
            epoch=np.array(history["epoch"]),
            train_loss=np.array(history["train_loss"]),
            train_acc=np.array(history["train_acc"]),
            val_loss=np.array(history["val_loss"]),
            val_acc=np.array(history["val_acc"]),
        )

        return student

    def _warm_start_mlp(
        self,
        student,
        X_train,
        teacher_probs,
    ):
        X = torch.tensor(
            X_train,
            dtype=torch.float32,
        )

        Y = torch.tensor(
            teacher_probs,
            dtype=torch.float32,
        )

        # -------------------------
        # Train / validation split
        # -------------------------
        val_split = 0.1

        n_samples = len(X)
        n_val = int(n_samples * val_split)

        indices = torch.randperm(n_samples)

        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_tr, Y_tr = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        train_loader = DataLoader(
            TensorDataset(X_tr, Y_tr),
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        )

        val_loader = DataLoader(
            TensorDataset(X_val, Y_val),
            batch_size=256,
            shuffle=False,
            pin_memory=True,
        )

        optimizer = torch.optim.AdamW(
            student.mlp.parameters(),
            lr=self.lr,
        )

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
        }

        warm_epochs = self.warm_epochs

        patience = self.patience
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        student.train()

        for epoch in range(warm_epochs):

            # -------------------------
            # Training
            # -------------------------
            train_loss = 0.0

            for x, y in train_loader:

                x = x.to(
                    self.device,
                    non_blocking=True,
                )

                y = y.to(
                    self.device,
                    non_blocking=True,
                )

                logits = student.mlp_forward(x)

                loss = F.kl_div(
                    F.log_softmax(
                        logits,
                        dim=1,
                    ),
                    y,
                    reduction="batchmean",
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # -------------------------
            # Validation
            # -------------------------
            student.eval()

            val_loss = 0.0

            with torch.no_grad():

                for x, y in val_loader:

                    x = x.to(
                        self.device,
                        non_blocking=True,
                    )

                    y = y.to(
                        self.device,
                        non_blocking=True,
                    )

                    logits = student.mlp_forward(x)

                    loss = F.kl_div(
                        F.log_softmax(
                            logits,
                            dim=1,
                        ),
                        y,
                        reduction="batchmean",
                    )

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            student.train()

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(
                f"Warm Start "
                f"{epoch+1:03d}/{warm_epochs} | "
                f"Train={train_loss:.6f} | "
                f"Val={val_loss:.6f}"
            )

            # -------------------------
            # Early stopping
            # -------------------------
            if val_loss < best_val_loss:

                best_val_loss = val_loss
                patience_counter = 0

                best_state = {
                    k: v.cpu().clone() for k, v in student.state_dict().items()
                }

            else:

                patience_counter += 1

                if patience_counter >= patience:

                    print(
                        f"Early stopping at epoch {epoch+1}. "
                        f"Best val loss={best_val_loss:.6f}"
                    )
                    break

        # Restore best validation model
        if best_state is not None:

            student.load_state_dict(best_state)

        # -------------------------
        # Save history
        # -------------------------
        history_dir = Path("checkpoints") / self.dataset_name

        history_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez(
            history_dir / "warm_training.npz",
            epoch=np.array(history["epoch"]),
            train_loss=np.array(history["train_loss"]),
            val_loss=np.array(history["val_loss"]),
        )

        return student

    # ------------------------------------------------
    # Main
    # ------------------------------------------------

    def fit(
        self,
        dataset_cls,
        config,
        seed=42,
    ):

        print_header("Distillation")

        if self.train_fidelity == -1:
            train_dataset = load_datasets(
                dataset_cls,
                preprocess_policy=config["preprocess_policy"],
                fidelity=self.train_fidelity,
                seed=seed,
            )
        else:
            train_dataset, _ = load_datasets(
                dataset_cls,
                preprocess_policy=config["preprocess_policy"],
                fidelity=self.train_fidelity,
                seed=seed,
            )

        print_step("Loading CNN encoder")

        encoder, _ = get_encoder(config["encoder"])

        encoder.to(self.device)

        print_step("Embedding train data")

        student = Student(
            self.encoder_name,
            hidden_dim=self.embedding_dim,
            num_classes=dataset_cls.num_classes,
        ).to(self.device)

        distill_times = {}

        if self.ditill_flag:

            start = time.perf_counter()

            X_train, y_train = self.embedder.embed(
                train_dataset,
            )

            distill_times["embedding_seconds"] = time.perf_counter() - start

            start = time.perf_counter()

            teacher, reducer = self._train_teacher(
                X_train,
                y_train,
                config,
                seed,
            )

            soft_labels = self._teacher_probs(
                teacher,
                reducer,
                X_train,
            )

            distill_times["teacher_predict_proba_seconds"] = time.perf_counter() - start

            start = time.perf_counter()
            student = self._warm_start_mlp(
                student=student, X_train=X_train, teacher_probs=soft_labels
            )
            distill_times["warm_start_mlp_seconds"] = time.perf_counter() - start

        else:
            teacher=None
            reducer=None

        student_train_dataset = dataset_cls(
            split="train",
            transform=build_transform(train=True),
        )

        student_val_dataset = dataset_cls(
            split="train",
            transform=build_transform(train=False),
        )

        start = time.perf_counter()

        student = self._train_student(
            student_train_dataset,
            student_val_dataset,
            student,
        )
        distill_times["student_train_seconds"] = time.perf_counter() - start

        # -------------------------
        # Save timing checkpoint
        # -------------------------
        checkpoint_dir = Path("checkpoints") / self.dataset_name
        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            checkpoint_dir / "distill_time.json",
            "w",
        ) as f:
            json.dump(
                distill_times,
                f,
                indent=4,
            )

        return teacher, student, reducer
