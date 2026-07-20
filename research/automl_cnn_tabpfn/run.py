import itertools
import subprocess

instances = [-1]
models = ["efficientnet_b0", "resnet18", "mobilenet", "densenet121"]
datasets = ["flowers", "fashion", "emotions"]
resize_sizes = [224, 0]  # 0 = original image size, no resizing
augment_flags = [0, 1]  # 0 = no augmentation, 1 = augmentation on

combinations = list(
    itertools.product(instances, models, datasets, resize_sizes, augment_flags)
)
print(f"Total runs planned: {len(combinations)}\n")

for ins, model, dataset, resize_size, augment in combinations:
    cmd = [
        "python",
        "train_and_test.py",
        "--dataset",
        dataset,
        "--number_instances",
        str(ins),
        "--model",
        model,
        "--resize_size",
        str(resize_size),
    ]

    if augment:
        cmd += ["--augment"]

    print("Running:", " ".join(cmd))

    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(
                f"✅ Finished dataset={dataset}, model={model}, instances={ins}, "
                f"resize={resize_size}, augment={augment}\n"
            )
        else:
            print(
                f"❌ Non-zero exit for dataset={dataset}, model={model}, instances={ins}, "
                f"resize={resize_size}, augment={augment}\n"
            )
    except Exception as e:
        print(
            f"❌ Failed for dataset={dataset}, model={model}, instances={ins}, "
            f"resize={resize_size}, augment={augment} — {e}\n"
        )
