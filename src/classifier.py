import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from src.automl.utils import print_header, print_step
from src.dataset.config import DATASETS
from src.dataset.loaders import load_test_datasets
from src.models.embedders import Embedder
from src.models.encoders import get_encoder
from src.models.reducers import PCAReducer
from src.models.student import Student
from src.models.tabpfn import TabPFNModel
from src.trainer.distill import Distiller
from src.trainer.utils import get_best_config, get_latest_config

class ImageClassifier:
    def __init__(
        self,
        dataset: str,
        train_fidelity: int = -1,
        seed: int = 42,
        batch_size=64,
        tabpfn_mode: str = "local",
        distill_flag: bool = False,
    ):

        self.DATASETS = DATASETS
        if dataset not in self.DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Available datasets: {list(self.DATASETS.keys())}"
            )

        self.dataset_name = dataset
        self.dataset_cls = self.DATASETS[dataset]
        self.num_classes = self.dataset_cls.num_classes
        self.seed = seed
        self.tabpfn_mode = tabpfn_mode
        self.train_fidelity = train_fidelity
        self.ditill_flag = distill_flag

        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.config = self.get_config()
        self.encoder_name = self.config["encoder"]

        self.init_paths()
        self.encoder, self.encoder_dim = get_encoder(self.config["encoder"])
        self.encoder.eval()
        self.encoder.to(self.device)
        self.batch_size = batch_size

        if self.config["embedding_dim"] != None:
            self.encoder_dim = self.config["embedding_dim"]

        self.tabpfn_model = TabPFNModel(
            mode=tabpfn_mode,
            seed=seed,
            device=self.device,
        )

        self.embedder = Embedder(self.encoder, device=self.device)
        self.student = Student(
            encoder_name=self.encoder_name,
            hidden_dim=self.encoder_dim,
            num_classes=self.dataset_cls.num_classes,
        )
        self.reducer = PCAReducer(dim=self.config["embedding_dim"], seed=self.seed)

        self.distiller = Distiller(
            train_fidelity=self.train_fidelity,
            seed=self.seed,
            tabpfn_model=self.tabpfn_model,
            embedder=self.embedder,
            device=self.device,
            dataset_name=self.dataset_name,
            encoder_name=self.encoder_name,
            embedding_dim=self.encoder_dim,
            ditill_flag=self.ditill_flag,
        )

    def init_paths(
        self,
    ):
        checkpoint_root = Path("checkpoints")
        self.folder = checkpoint_root / f"{self.dataset_name}"

        self.folder.mkdir(parents=True, exist_ok=True)

        self.student_path = self.folder / "student.pt"
        self.teacher_path = self.folder / "teacher.pkl"
        self.reducer_path = self.folder / "reducer.pkl"

    def get_config(
        self,
    ):
        configs = get_latest_config(self.dataset_name)
        best_config = get_best_config(configs)["config"]
        print(best_config)
        return best_config

    def fit(
        self,
    ):
        self.teacher, self.student, self.reducer = self.distiller.fit(
            self.dataset_cls, self.config, seed=self.seed
        )
        self.save_checkpoints()

    def save_checkpoints(
        self,
    ):
        torch.save(self.student.state_dict(), self.student_path)
        return True

    def load_checkpoints(self):
        loaded = False

        # Load student weights
        if os.path.exists(self.student_path):
            state_dict = torch.load(self.student_path, map_location=self.device)
            self.student.load_state_dict(state_dict)
            self.student.to(self.device)
            self.student.eval()
            loaded = True

        return loaded

    def predict(self, dataset_cls) -> Tuple[np.ndarray, np.ndarray]:

        if not self.load_checkpoints():
            raise RuntimeError("Failed to load checkpoints.")

        dataset = load_test_datasets(
            dataset_cls,
            seed=self.seed,
        )

        self.student.eval()

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        predictions = []
        labels = []

        with torch.no_grad():

            for batch in loader:

                images = batch[0].to(self.device)

                logits = self.student(images)

                pred = torch.argmax(logits, dim=1)

                predictions.append(pred.cpu().numpy())

                if len(batch) > 1:
                    labels.append(batch[1].numpy())

        predictions = np.concatenate(predictions)

        if len(labels) > 0:
            labels = np.concatenate(labels)
        else:
            labels = None

        self.save_predictions(predictions)

        return predictions, labels

    def save_predictions(self, predictions):
        """
        Save predictions for checkpoints and benchmark evaluation.
        """

        # Save inside checkpoint directory
        self.folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        prediction_path = self.folder / "predictions.npy"

        with prediction_path.open("wb") as f:
            np.save(f, predictions)

        print_step(f"Predictions saved to {prediction_path}")

        # Save exam submission file
        if self.dataset_name == "skin_cancer":

            test_output_path = Path("data/exam_dataset/predictions.npy")

            test_output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with test_output_path.open("wb") as f:
                np.save(f, predictions)

            final_output_path = Path("final_test_preds.npy")

            with final_output_path.open("wb") as f:
                np.save(f, predictions)

            print_step(f"Exam predictions saved to {test_output_path}")

        return prediction_path

    def evaluate(self):
        print_header("Evaluation")

        if not self.load_checkpoints():
            return False

        # -------------------------
        # Teacher
        # -------------------------
        teacher_predictions = None
        teacher_accuracy = None

        if self.ditill_flag:

            if self.teacher is not None:

                test_dataset = load_test_datasets(
                    self.dataset_cls,
                    seed=self.seed,
                )

                print_step("Embedding test data")

                X_test, y_test = self.embedder.embed(test_dataset)

                print_step("Teacher prediction")

                teacher_predictions = self.distiller._teacher_predict(
                    self.teacher,
                    self.reducer,
                    X_test,
                )

                if y_test is not None:
                    teacher_accuracy = float((teacher_predictions == y_test).mean())
                    print(f"Teacher Accuracy: {teacher_accuracy:.4f}")

        # -------------------------
        # Student
        # -------------------------
        print_step("Student prediction")

        student_predictions, labels = self.predict(self.dataset_cls)

        student_accuracy = None

        if labels is not None:
            student_accuracy = float((student_predictions == labels).mean())
            print(f"Student Accuracy: {student_accuracy:.4f}")

        # -------------------------
        # Save
        # -------------------------

        results = {
            "config": self.config,
            "teacher_accuracy": teacher_accuracy,
            "student_accuracy": student_accuracy,
        }

        results_path = self.folder / "results.json"

        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        print_step(f"Results saved to {results_path}")

        return results


if __name__ == "__main__":
    clf = ImageClassifier(
        "skin_cancer",
        tabpfn_mode="local",
    )

    clf.fit()
    # clf.load_checkpoints()
    clf.evaluate()
