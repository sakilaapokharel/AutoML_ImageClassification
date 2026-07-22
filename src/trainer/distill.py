# trainer/distill.py

import json
import pickle

from pathlib import Path
from datetime import datetime

import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split,
    ConcatDataset
)

from torchvision import transforms


from models.encoders import get_encoder
from models.reducers import PCAReducer
from models.student import Student

from dataset.samplers import InstancesPerClassDataset
from dataset.loaders import TransformDataset

from models.tabpfn import TabPFNModel

from automl.utils import (
    print_header,
    print_step,
    print_progress,
    ResizeToMultipleOf14
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Distiller:

    def __init__(
        self,
        batch_size=64,
        epochs=111,
        lr=1e-3,
        alpha=0.3,
        temperature=2.0,
        patience=29,
        val_ratio=0.1,
        device="cuda",
        checkpoint_root="checkpoints",
        fidelity=56,
        seed=42,
        tabpfn_model=TabPFNModel,
        finetune=False,
        finetune_epochs=111,
        finetune_lr=1e-4,
        finetune_patience=29,
    ):

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        self.alpha = alpha
        self.temperature = temperature

        self.patience = patience
        self.val_ratio = val_ratio

        # supervised fine-tuning phase, run after distillation on the same
        # train/val split, using only true hard labels (no KD term), at a
        # lower LR -- sharpens the decision boundary the soft-label training
        # may have blurred. Set finetune=False to skip and keep old behavior.
        self.finetune = finetune
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.finetune_patience = finetune_patience

        self.device = device

        self.checkpoint_root = Path(checkpoint_root)
        self.fidelity = fidelity
        
        self.seed = seed
        self.tabpfn_model = tabpfn_model

    # ------------------------------------------------
    # Transforms
    # ------------------------------------------------

    def _get_transform(
        self,
        config,
    ):

        base = []

        if config["resize"] is not None:

            base.append(
                transforms.Resize(
                    (
                        config["resize"],
                        config["resize"],
                    )
                )
            )

        else:

            base.append(
                ResizeToMultipleOf14()
            )

        if config["augmentation"] == "randaugment":

            train_transform = transforms.Compose(
                base
                + [
                    transforms.RandAugment(
                        num_ops=2,
                        magnitude=5,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=IMAGENET_MEAN,
                        std=IMAGENET_STD,
                    ),
                ]
            )

        else:

            train_transform = transforms.Compose(
                base
                + [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=IMAGENET_MEAN,
                        std=IMAGENET_STD,
                    ),
                ]
            )

        original_transform = transforms.Compose(
            base
            + [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

        test_transform = transforms.Compose(
            base
            + [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

        return (
            original_transform,
            train_transform,
            test_transform,
        )


    # ------------------------------------------------
    # Dataset
    # ------------------------------------------------

    def _load_datasets(
        self,
        dataset_cls,
        config,
        fidelity,
        seed,
    ):

        (
            original_transform,
            train_transform,
            test_transform,
        ) = self._get_transform(config)

        print_step("Loading raw training dataset")

        raw_train = dataset_cls(
            split="train",
            transform=None,
            download=False,
        )

        #
        # Fidelity selection
        #
        if fidelity != -1:

            sampled_train = InstancesPerClassDataset(
                raw_train,
                instances_per_class=fidelity,
                seed=seed,
            )

        else:

            sampled_train = raw_train

        #
        # Original images
        #
        original_dataset = TransformDataset(
            sampled_train,
            original_transform,
        )

        #
        # Original + RandAugment
        #
        if config["augmentation"] == "randaugment":

            augmented_dataset = TransformDataset(
                sampled_train,
                train_transform,
            )

            train_dataset = ConcatDataset(
                [
                    original_dataset,
                    augmented_dataset,
                ]
            )

        else:

            train_dataset = original_dataset

        #
        # Test
        #
        test_dataset = dataset_cls(
            split="test",
            transform=test_transform,
            download=False,
        )

        return (
            train_dataset,
            test_dataset,
        )
    # ------------------------------------------------
    # CNN embedding
    # ------------------------------------------------

    def _embed(
        self,
        encoder,
        dataset,
    ):

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            pin_memory=True,
        )

        embeddings = []
        labels = []

        total = len(dataset)

        seen = 0

        encoder.eval()

        with torch.no_grad():

            for x, y in loader:

                x = x.to(self.device)

                z = encoder(x)

                embeddings.append(z.cpu().numpy())

                labels.append(y.numpy())

                seen += len(x)

                print_progress(
                    seen,
                    total,
                    every=500,
                )

        return (
            np.concatenate(embeddings),
            np.concatenate(labels),
        )

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


        # TabPFNModel.predict_proba already batches internally (see
        # models/tabpfn.py) -- delegate to it instead of chunking again here.
        return teacher.predict_proba(
            X,
        )

    # ------------------------------------------------
    # Student
    # ------------------------------------------------

    def _train_student(
        self,
        X_train,
        teacher_probs,
        labels,
        num_classes,
    ):

        print_step("Training student")

        dataset = TensorDataset(
            torch.tensor(
                X_train,
                dtype=torch.float32,
            ),
            torch.tensor(
                teacher_probs,
                dtype=torch.float32,
            ),
            torch.tensor(
                labels,
                dtype=torch.long,
            ),
        )

        val_size = int(len(dataset) * self.val_ratio)

        train_size = len(dataset) - val_size

        train_ds, val_ds = random_split(
            dataset,
            [
                train_size,
                val_size,
            ],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
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
    # Fine-tune (supervised, post-distillation)
    # ------------------------------------------------

    def _finetune_student(
        self,
        student,
        train_loader,
        val_loader,
    ):

        print_step("Fine-tuning student on true labels")

        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=self.finetune_lr,
            weight_decay=1e-4,
        )

        ce = nn.CrossEntropyLoss()

        best_loss = float("inf")

        best_state = None

        patience_counter = 0

        for epoch in range(self.finetune_epochs):

            student.train()

            train_loss = 0

            # train_loader/val_loader yield (x, soft, y) triples from the
            # distillation TensorDataset -- soft labels are simply unused here.
            for x, _soft, y in train_loader:

                x = x.to(self.device)

                y = y.to(self.device)

                logits = student(x)

                loss = ce(
                    logits,
                    y,
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # validation

            student.eval()

            val_loss = 0

            correct = 0

            total = 0

            with torch.no_grad():

                for x, _soft, y in val_loader:

                    x = x.to(self.device)

                    y = y.to(self.device)

                    logits = student(x)

                    loss = ce(
                        logits,
                        y,
                    )

                    val_loss += loss.item()

                    pred = torch.argmax(logits, dim=1)

                    correct += (pred == y).sum().item()

                    total += len(y)

            val_loss /= len(val_loader)

            val_acc = correct / total

            print(
                f"[Finetune] Epoch {epoch+1}/{self.finetune_epochs} "
                f"Train={train_loss:.4f} "
                f"Val={val_loss:.4f} "
                f"ValAcc={val_acc:.4f}"
            )

            if val_loss < best_loss:

                best_loss = val_loss

                best_state = student.state_dict()

                patience_counter = 0

            else:

                patience_counter += 1

            if patience_counter >= self.finetune_patience:

                print("Early stopping (fine-tune)")

                break

        if best_state:

            student.load_state_dict(best_state)

        return student

    # ------------------------------------------------
    # Save
    # ------------------------------------------------

    def _save(
        self,
        dataset,
        config,
        student,
        reducer,
        teacher=None,
        results=None,
        finetuned=False,
    ):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        folder = self.checkpoint_root / f"{dataset}_{timestamp}"

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        # save student weights
        student_path = folder / "student.pt"

        torch.save(
            student.state_dict(),
            student_path,
        )

        # save PCA
        pca_path = None

        if reducer is not None:

            pca_path = folder / "pca.pkl"

            with open(
                pca_path,
                "wb",
            ) as f:

                pickle.dump(
                    reducer,
                    f,
                )

        # save teacher
        teacher_path = None

        if teacher is not None:

            teacher_path = folder / "teacher.pkl"

            with open(
                teacher_path,
                "wb",
            ) as f:

                pickle.dump(
                    teacher,
                    f,
                )

        # save config + paths

        checkpoint = {
            "student": str(student_path),
            "pca": (str(pca_path) if pca_path else None),
            "teacher": (str(teacher_path) if teacher_path else None),
            "config": config,
            "created": timestamp,
            "results": (results if results else None),
            "folder": str(folder),
            "finetuned": finetuned,
            "n_estimators" : self.fidelity,
            "batch":self.batch_size
        }

        with open(
            folder / "checkpoint.json",
            "w",
        ) as f:

            json.dump(
                checkpoint,
                f,
                indent=4,
            )

        print(f"Saved checkpoint: {folder}")

        return checkpoint

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

        train_dataset, test_dataset = self._load_datasets(
            dataset_cls,
            config,
            fidelity=self.fidelity,
            seed=seed
        )

        print_step("Loading CNN encoder")

        encoder, _ = get_encoder(config["encoder"])

        encoder.to(self.device)

        print_step("Embedding train data")

        X_train, y_train = self._embed(
            encoder,
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
            dataset_cls.num_classes,
        )

        if self.finetune:

            student = self._finetune_student(
                student,
                train_loader,
                val_loader,
            )

        checkpoint = self._save(
            dataset_cls._dataset_name,
            config,
            student,
            reducer,
            teacher=teacher,
            finetuned=self.finetune,
        )

        print_step("Evaluating teacher and student")

        results = self.evaluate(
            dataset_cls,
            checkpoint,
        )

        return {
            "student": student,
            "checkpoint": checkpoint,
            "teacher_accuracy": results["teacher_accuracy"],
            "student_accuracy": results["student_accuracy"],
        }

    def _load_checkpoint(
        self,
        checkpoint,
    ):

        # with open(
        #     checkpoint_path,
        #     "r",
        # ) as f:

        #     checkpoint = json.load(f)

        reducer = None

        if checkpoint["pca"] is not None:

            with open(
                checkpoint["pca"],
                "rb",
            ) as f:

                reducer = pickle.load(f)

        teacher = None

        if checkpoint.get("teacher") is not None:

            with open(
                checkpoint["teacher"],
                "rb",
            ) as f:

                teacher = pickle.load(f)

        return (
            checkpoint,
            reducer,
            teacher,
        )

    def evaluate(
        self,
        dataset_cls,
        checkpoint_path,
    ):

        print_header("Testing Student")

        checkpoint, reducer, teacher = self._load_checkpoint(checkpoint_path)

        config = checkpoint["config"]

        # -------------------------
        # Encoder
        # -------------------------

        print_step("Loading encoder")

        encoder, _ = get_encoder(config["encoder"])

        encoder.to(self.device)

        encoder.eval()

        # -------------------------
        # Test dataset
        # -------------------------

        _, test_dataset = self._load_datasets(
            dataset_cls,
            config,
            fidelity=self.fidelity,
            seed=self.seed
        )

        print_step("Embedding test images")

        X_test, y_test = self._embed(
            encoder,
            test_dataset,
        )

        # -------------------------
        # Teacher accuracy
        # -------------------------

        teacher_accuracy = None

        if teacher is not None:

            print_step("Evaluating teacher")

            teacher_probs = self._teacher_probs(
                teacher,
                reducer,
                X_test,
            )

            teacher_preds = np.argmax(
                teacher_probs,
                axis=1,
            )

            teacher_accuracy = float(
                (teacher_preds == y_test).mean()
            )

            print(f"Teacher Accuracy: {teacher_accuracy:.4f}")

        # -------------------------
        # PCA
        # -------------------------

        if reducer is not None:

            print_step("Applying PCA")

            X_test = reducer.transform(X_test)

        # -------------------------
        # Student
        # -------------------------

        print_step("Loading student")

        student = Student(
            embedding_dim=X_test.shape[1],
            num_classes=dataset_cls.num_classes,
        )

        student.load_state_dict(
            torch.load(
                checkpoint["student"],
                map_location=self.device,
            )
        )

        student.to(self.device)

        student.eval()

        X_test = torch.tensor(
            X_test,
            dtype=torch.float32,
        ).to(self.device)

        y_test = torch.tensor(
            y_test,
            dtype=torch.long,
        ).to(self.device)

        # -------------------------
        # Accuracy
        # -------------------------

        correct = 0
        total = 0

        with torch.no_grad():

            for i in range(
                0,
                len(X_test),
                self.batch_size,
            ):

                x = X_test[i : i + self.batch_size]

                y = y_test[i : i + self.batch_size]

                logits = student(x)

                pred = torch.argmax(
                    logits,
                    dim=1,
                )

                correct += (pred == y).sum().item()

                total += len(y)

        student_accuracy = correct / total

        print(f"\nStudent Accuracy: {student_accuracy:.4f}")

        # -------------------------
        # Persist results
        # -------------------------

        results = {
            "teacher_accuracy": teacher_accuracy,
            "student_accuracy": student_accuracy,
        }

        checkpoint["results"] = results

        if checkpoint.get("folder") is not None:

            with open(
                Path(checkpoint["folder"]) / "checkpoint.json",
                "w",
            ) as f:

                json.dump(
                    checkpoint,
                    f,
                    indent=4,
                )

        return results

