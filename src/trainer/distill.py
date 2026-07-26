# trainer/distill.py

import json
import pickle

from pathlib import Path
from datetime import datetime

import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset

from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Subset

from models.encoders import get_encoder
from models.reducers import PCAReducer
from models.student import Student
from models.embedders import Embedder
from dataset.loaders import load_datasets

from models.tabpfn import TabPFNModel
from dataset.transforms import build_transform, TransformDataset
from dataset.datasets import StudentDistillDataset

from automl.utils import print_header, print_step, print_progress

from models.embedders import Embedder

class Distiller:

    def __init__(
        self,
        batch_size=32,
        epochs=200,
        lr=1e-3,
        alpha=0.7,
        temperature=4.0,
        patience=29,
        val_ratio=0.2,
        device="cuda",
        checkpoint_root="checkpoints",
        train_fidelity=-1,
        seed=42,
        tabpfn_model=TabPFNModel,
        embedder=Embedder,
    ):

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

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

    def load_dataset_for_training_student(
        self,
        X,
        teacher_probs,
        labels,
        batch_size,
        val_ratio,
        seed,
    ):

        indices = list(range(len(X)))

        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            stratify=labels,
            random_state=seed,
        )

        train_ds = StudentDistillDataset(
            X,
            teacher_probs,
            labels,
            train_idx,
        )

        val_ds = StudentDistillDataset(
            X,
            teacher_probs,
            labels,
            val_idx,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
        )

        return train_loader, val_loader

    def _train_student(
        self,
        X_train,
        teacher_probs,
        labels,
        config,
        reducer,
        num_classes,
    ):

        print_step("Training student")
        if reducer is not None:
            X_train = reducer.transform(X_train)

        train_loader, val_loader = self.load_dataset_for_training_student(
            X=X_train,
            teacher_probs=teacher_probs,
            labels=labels,
            batch_size=self.batch_size,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )

        student = Student(
            embedding_dim=X_train.shape[1],
            num_classes=num_classes,
        )

        student.to(self.device)

        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

        ce = nn.CrossEntropyLoss()

        best_loss = float("inf")

        best_state = None

        patience_counter = 0

        for epoch in range(self.epochs):

            student.train()

            train_loss = 0

            for x, soft, y in train_loader:

                x = x.to(self.device)

                soft = soft.to(self.device)

                y = y.to(self.device)

                logits = student(x)

                loss_ce = ce(
                    logits,
                    y,
                )

                loss_kd = F.kl_div(
                    F.log_softmax(
                        logits / self.temperature,
                        dim=1,
                    ),
                    F.softmax(
                        soft / self.temperature,
                        dim=1,
                    ),
                    reduction="batchmean",
                )

                loss = (
                    self.alpha * loss_ce
                    + (1 - self.alpha) * self.temperature**2 * loss_kd
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # validation

            student.eval()

            val_loss = 0

            with torch.no_grad():

                for x, soft, y in val_loader:

                    x = x.to(self.device)

                    soft = soft.to(self.device)

                    y = y.to(self.device)

                    logits = student(x)

                    loss_ce = ce(
                        logits,
                        y,
                    )

                    loss_kd = F.kl_div(
                        F.log_softmax(
                            logits / self.temperature,
                            dim=1,
                        ),
                        F.softmax(
                            soft / self.temperature,
                            dim=1,
                        ),
                        reduction="batchmean",
                    )

                    loss = (
                        self.alpha * loss_ce
                        + (1 - self.alpha) * self.temperature**2 * loss_kd
                    )

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(
                f"Epoch {epoch+1}/{self.epochs} "
                f"Train={train_loss:.4f} "
                f"Val={val_loss:.4f}"
            )

            if val_loss < best_loss:

                best_loss = val_loss

                best_state = student.state_dict()

                patience_counter = 0

            else:

                patience_counter += 1

            if patience_counter >= self.patience:

                print("Early stopping")

                break

        if best_state:

            student.load_state_dict(best_state)

        return student, train_loader, val_loader

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

        X_train, y_train = self.embedder.embed(
            train_dataset,
        )

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

        student, train_loader, val_loader = self._train_student(
            X_train,
            soft_labels,
            y_train,
            config,
            reducer,
            dataset_cls.num_classes,
        )

        if self.finetune:

            student = self._finetune_student(
                student,
                train_loader,
                val_loader,
            )

        return teacher, student, reducer

