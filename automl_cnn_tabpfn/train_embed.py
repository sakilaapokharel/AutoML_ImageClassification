import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

torch.backends.mps.is_available = lambda: False

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

from configs import DATASET_DIRS, MODEL_CHOICES
from pipeline import extract_dataset_features
from distill import MLPHead


def batched_get_embeddings(embedder, X_context, y_context, X_query, batch_size=200):
    """Returns embeddings shaped (n_estimators, n_samples, embed_dim)."""
    outputs = []
    for i in range(0, len(X_query), batch_size):
        chunk = embedder.get_embeddings(X_context, y_context, X_query[i:i + batch_size], data_source="test")
        outputs.append(chunk)
    return np.concatenate(outputs, axis=1)


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
                         help="max rows kept as TabPFN's context")
    parser.add_argument("--embed_batch_size", type=int, default=200,
                         help="batch size for TabPFN embedding extraction calls")
    args = parser.parse_args()

    print("Extracting CNN features...")
    X_train, y_train, X_test, y_test, embed_dim, randaug_magnitude = extract_dataset_features(
        DATASET_DIRS[args.dataset], args.model, args.number_instances,
        seed=args.seed, batch_size=args.batch_size, resize_size=args.resize_size, augment=args.augment,
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

    rng = np.random.default_rng(args.seed)
    if len(X_train) > args.tabpfn_max_train:
        idx = rng.choice(len(X_train), size=args.tabpfn_max_train, replace=False)
        X_context, y_context = X_train[idx], y_train[idx]
        print(f"TabPFN context capped: {len(X_train)} -> {args.tabpfn_max_train} rows")
    else:
        X_context, y_context = X_train, y_train

    print("Building TabPFN embedder (frozen)...")
    from tabpfn_extensions import TabPFNClassifier
    from tabpfn_extensions.embedding import TabPFNEmbedding

    clf = TabPFNClassifier(n_estimators=1, device="mps")
    embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)
    embedder.fit(X_context, y_context)

    print("Extracting TabPFN embeddings for train pool...")
    train_embed = batched_get_embeddings(embedder, X_context, y_context, X_train, batch_size=args.embed_batch_size)
    train_embed = train_embed.mean(axis=0)

    print("Extracting TabPFN embeddings for test set...")
    test_embed = batched_get_embeddings(embedder, X_context, y_context, X_test, batch_size=args.embed_batch_size)
    test_embed = test_embed.mean(axis=0)

    num_classes = len(np.unique(y_train))
    print(f"TabPFN embedding dim: {train_embed.shape[1]} | num_classes: {num_classes}")

    print("Training MLP on frozen TabPFN embeddings (only MLP trainable)...")
    mlp = MLPHead(in_dim=train_embed.shape[1], num_classes=num_classes)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    X_tr = torch.tensor(train_embed, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)

    mlp.train()
    for epoch in range(args.mlp_epochs):
        optimizer.zero_grad()
        logits = mlp(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch {epoch}: loss {loss.item():.4f}")

    mlp.eval()
    with torch.no_grad():
        test_logits = mlp(torch.tensor(test_embed, dtype=torch.float32))
        test_pred = test_logits.argmax(dim=1).numpy()
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\n=== CNN + TabPFN-embedding + MLP: {args.dataset} | {args.model} | N/class={args.number_instances} ===")
    print(f"Test accuracy: {test_acc:.4f}")
    print(classification_report(y_test, test_pred, zero_division=0))

    save_dir = Path("saved_models_embed")
    save_dir.mkdir(exist_ok=True)
    tag = f"{args.dataset}_{args.model}_n{args.number_instances}_aug{int(args.augment)}"

    torch.save(mlp.state_dict(), save_dir / f"{tag}_mlp.pt")
    if pca is not None:
        with open(save_dir / f"{tag}_pca.pkl", "wb") as f:
            pickle.dump(pca, f)
    np.savez(save_dir / f"{tag}_tabpfn_context.npz", X_context=X_context, y_context=y_context)

    with open(save_dir / f"{tag}_meta.json", "w") as f:
        json.dump({
            "model_name": args.model,
            "in_dim": train_embed.shape[1],
            "num_classes": num_classes,
            "resize_size": args.resize_size,
            "used_pca": pca is not None,
            "test_accuracy": test_acc,
        }, f, indent=2)

    results_dir = Path("results_embed")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / f"{tag}.json", "w") as f:
        json.dump({
            "dataset": args.dataset, "model": args.model, "n_per_class": args.number_instances,
            "augment": args.augment, "resize_size": args.resize_size,
            "tabpfn_max_train": args.tabpfn_max_train, "test_accuracy": test_acc,
        }, f, indent=2)

    print(f"Saved to {save_dir}/{tag}_* and {results_dir}/{tag}.json")


if __name__ == "__main__":
    main()