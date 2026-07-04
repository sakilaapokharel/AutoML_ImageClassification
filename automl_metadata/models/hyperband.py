import math
import os
import time
import pandas as pd
from automl_metadata.utils import merge_all_csv


def hyperband_search(
    config_sampler,
    run_config,
    meta_features=None,
    max_epochs=27,
    eta=3,
    output_dir="metadata",
    verbose=True,
):
    """
    Full Hyperband with:
    - Successive halving
    - Live CSV logging (each run saved immediately)
    - Full schema row (meta + config + results)
    """
    log_dir = f"{output_dir}/individual_logs"
    os.makedirs(log_dir, exist_ok=True)
    run_id = int(time.time())
    log_path = os.path.join(log_dir, f"hyperband_{run_id}.csv")

    if meta_features is None:
        meta_features = {}

    s_max = int(math.log(max_epochs, eta))
    B = (s_max + 1) * max_epochs

    if verbose:
        print("\n[Hyperband START]")
        print(f"max_epochs={max_epochs}, eta={eta}, s_max={s_max}")
        print(f"log_file={log_path}\n")

    for s in reversed(range(s_max + 1)):

        n = int(math.ceil(B / max_epochs / (s + 1) * (eta**s)))
        r = max_epochs * (eta ** (-s))

        configs = [config_sampler() for _ in range(n)]

        if verbose:
            print(f"\n[Bracket s={s}] configs={n}, init_r={int(r)}")

        for i in range(s + 1):

            n_i = int(n * (eta ** (-i)))
            r_i = max(1, int(round(r * (eta**i))))

            evaluated = []

            if verbose:
                print(f"\n  [Rung {i}] epochs={r_i}, configs={len(configs)}")

            for cfg in configs:

                if verbose:
                    print(f"    Training: {cfg}")

                val_acc, extra = run_config(cfg, r_i)

                # ---------------------------------------
                # FULL ROW (FINAL SCHEMA MATCHING YOUR DF)
                # ---------------------------------------
                row = dict(meta_features)
                row.update(
                    {
                        "model": cfg["model"],
                        "augmentation": cfg["augmentation"],
                        "loss": cfg["loss"],
                        "hyperband_bracket": s,
                        "hyperband_rung": i,
                        "epochs_trained": r_i,
                        "hyperband_val_accuracy": val_acc,
                        "test_accuracy": extra.get("test_accuracy"),
                        "compute_time_sec": extra.get("compute_time_sec"),
                    }
                )

                evaluated.append((cfg, val_acc))
                if verbose:
                    print(f"      val_acc={val_acc:.4f}")

                # ---------------------------------------
                # LIVE SAVE (INDIVIDUAL LOG FILE)
                # ---------------------------------------
                df_row = pd.DataFrame([row])
                df_row.to_csv(
                    log_path,
                    mode="a",
                    header=not os.path.exists(log_path),
                    index=False,
                )

            # ---------------------------------------
            # SUCCESSIVE HALVING SELECTION
            # ---------------------------------------
            evaluated.sort(key=lambda x: x[1], reverse=True)
            keep = max(1, int(len(configs) / eta))
            configs = [c for c, _ in evaluated[:keep]]

            if verbose:
                print(f"  Kept {len(configs)} configs")

    if verbose:
        print("\n[Hyperband DONE]")
        print(f"Saved → {log_path}")

    # Better readeability for user
    # print(output_dir)
    # merge_all_csv(output_dir)

    return log_path
