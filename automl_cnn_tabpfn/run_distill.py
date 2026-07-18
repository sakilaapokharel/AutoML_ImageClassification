import itertools
import subprocess

instances = [11, 29]
models = ["resnet18", "efficientnet_b0", "mobilenet", "densenet121"]
datasets = ["flowers", "fashion", "emotions"]
augment_flags = [0, 1]

combinations = list(itertools.product(instances, models, datasets, augment_flags))
print(f"Total distillation runs planned: {len(combinations)}\n")

for ins, model, dataset, augment in combinations:
    cmd = [
        "python",
        "distill_and_test.py",
        "--dataset", dataset,
        "--number_instances", str(ins),
        "--model", model,
    ]
    if augment:
        cmd += ["--augment"]

    print("Running:", " ".join(cmd))

    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"✅ Finished dataset={dataset}, model={model}, instances={ins}, augment={augment}\n")
        else:
            print(f"❌ Non-zero exit for dataset={dataset}, model={model}, instances={ins}, augment={augment}\n")
    except Exception as e:
        print(f"❌ Failed for dataset={dataset}, model={model}, instances={ins}, augment={augment} — {e}\n")