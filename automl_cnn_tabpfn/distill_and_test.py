import argparse
import json
from pathlib import Path

import pickle
import numpy as np
import torch
from dotenv import load_dotenv
load_dotenv()

torch.backends.mps.is_available = lambda: False  # force true CPU, no silent MPS fallback

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

from configs import DATASET_DIRS, MODEL_CHOICES
from pipeline import extract_dataset_features
from distill import train_mlp_head


def batched_predict_proba(clf, X, batch_size=200):
    outputs = []
    for i in range(0, len(X), batch_size):
        outputs.append(clf.predict_proba(X[i:i + batch_size]))
    return np.vstack(outputs)


def batched_predict(clf, X, batch_size=200):
    outputs = []
    for i in range(0, len(X), batch_size):
        outputs.append(clf.predict(X[i:i + batch_size]))
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_DIRS.keys()))
    parser.add_argument("--number_instances", type=int, required=True)
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--pca_dim", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--mlp_epochs", type=int, default=200)
    parser.add_argument("--tabpfn_max_train", type=int, default=2000,
                         help="max rows TabPFN itself trains on — subsampled from X_train if exceeded. "
                              "The MLP student still uses the FULL X_train regardless.")
    parser.add_argument("--tabpfn_predict_batch", type=int, default=200,
                         help="batch size for TabPFN predict/predict_proba calls, to avoid OOM on large test sets")
    args = parser.parse_args()

    print("Extracting CNN features...")
    X_train, y_train, X_test, y_test, embed_dim, randaug_magnitude = extract_dataset_features(
        DATASET_DIRS[args.dataset],
        args.model,
        args.number_instances,
        seed=args.seed,
        batch_size=args.batch_size,
        resize_size=args.resize_size,
        augment=args.augment,
    )

    pca = None
    if args.pca_dim:
        max_components = min(X_train.shape[0], X_train.shape[1])
        effective_dim = min(args.pca_dim, max_components)
        if effective_dim < X_train.shape[1]:
            pca = PCA(n_components=effective_dim, random_state=args.seed)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            print(f"PCA applied: {embed_dim} -> {effective_dim} dims")

    # cap rows TabPFN itself trains on — MLP still sees full X_train
    rng = np.random.default_rng(args.seed)
    if len(X_train) > args.tabpfn_max_train:
        idx = rng.choice(len(X_train), size=args.tabpfn_max_train, replace=False)
        X_train_tabpfn, y_train_tabpfn = X_train[idx], y_train[idx]
        print(f"TabPFN train set capped: {len(X_train)} -> {args.tabpfn_max_train} rows")
    else:
        X_train_tabpfn, y_train_tabpfn = X_train, y_train

    print("Fitting local TabPFN (teacher)...")
    from tabpfn_client import TabPFNClassifier
    clf = TabPFNClassifier()
    clf.fit(X_train_tabpfn, y_train_tabpfn)

    # teacher's own performance, for comparison
    tabpfn_test_pred = batched_predict(clf, X_test, batch_size=args.tabpfn_predict_batch)
    tabpfn_test_acc = accuracy_score(y_test, tabpfn_test_pred)

    # teacher's soft labels on the FULL training pool (not the capped subset)
    print("Getting TabPFN soft labels on the training pool...")
    teacher_probs = batched_predict_proba(clf, X_train, batch_size=args.tabpfn_predict_batch)

    num_classes = teacher_probs.shape[1]
    print(f"Training MLP student (num_classes={num_classes})...")
    mlp = train_mlp_head(X_train, y_train, teacher_probs, num_classes, epochs=args.mlp_epochs)

    mlp.eval()
    with torch.no_grad():
        mlp_logits = mlp(torch.tensor(X_test, dtype=torch.float32))
        mlp_pred = mlp_logits.argmax(dim=1).numpy()
    mlp_test_acc = accuracy_score(y_test, mlp_pred)

    print(f"\n=== Comparison: {args.dataset} | {args.model} | N/class={args.number_instances} ===")
    print(f"TabPFN (teacher) test accuracy: {tabpfn_test_acc:.4f}")
    print(f"MLP (student)   test accuracy: {mlp_test_acc:.4f}")
    print(f"Gap (teacher - student):       {tabpfn_test_acc - mlp_test_acc:.4f}")
    print("\nMLP classification report:")
    print(classification_report(y_test, mlp_pred, zero_division=0))

    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    tag = f"{args.dataset}_{args.model}_n{args.number_instances}_aug{int(args.augment)}"

    torch.save(mlp.state_dict(), save_dir / f"{tag}_mlp.pt")

    if pca is not None:
        with open(save_dir / f"{tag}_pca.pkl", "wb") as f:
            pickle.dump(pca, f)

    with open(save_dir / f"{tag}_meta.json", "w") as f:
        json.dump({
            "model_name": args.model,
            "in_dim": X_train.shape[1],
            "num_classes": num_classes,
            "resize_size": args.resize_size,
            "used_pca": pca is not None,
        }, f, indent=2)

    print(f"Saved MLP model + metadata to {save_dir}/{tag}_*")

    out_dir = Path("results_distill")
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.dataset}_{args.model}_n{args.number_instances}_aug{int(args.augment)}"
    out_path = out_dir / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "model": args.model,
            "n_per_class": args.number_instances,
            "augment": args.augment,
            "resize_size": args.resize_size,
            "tabpfn_max_train": args.tabpfn_max_train,
            "tabpfn_test_acc": tabpfn_test_acc,
            "mlp_test_acc": mlp_test_acc,
            "gap": tabpfn_test_acc - mlp_test_acc,
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()