def get_best_run(results):

    runs = results.get_all_runs()

    # Remove failed runs
    valid_runs = [r for r in runs if r.loss is not None]

    if len(valid_runs) == 0:
        raise RuntimeError("No valid BOHB runs found")

    # BOHB minimizes loss
    best_run = min(valid_runs, key=lambda r: r.loss)

    return best_run


def extract_best_run(results):

    best_run = get_best_run(results)

    return {
        "loss": float(best_run.loss),
        "budget": float(best_run.budget),
        "accuracy": float(best_run.info.get("val_accuracy", 0)),
        "compute_time_sec": float(best_run.info.get("compute_time_sec", 0)),
        "config": {str(k): v for k, v in best_run.config.items()},
    }


def extract_best_portfolio(portfolio_results):

    best = None

    for idx, portfolio in enumerate(portfolio_results):

        results = portfolio["results"]

        best_run = get_best_run(results)

        candidate = {
            "portfolio_index": idx,
            "portfolio_config": portfolio["portfolio_config"],
            "loss": float(best_run.loss),
            "budget": float(best_run.budget),
            "accuracy": float(best_run.info.get("val_accuracy", 0)),
            "compute_time_sec": float(best_run.info.get("compute_time_sec", 0)),
            "config": {str(k): v for k, v in best_run.config.items()},
        }

        if best is None or candidate["accuracy"] > best["accuracy"]:

            best = candidate

    return best
