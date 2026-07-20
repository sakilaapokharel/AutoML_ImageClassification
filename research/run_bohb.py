# run_bohb.py

import argparse
import json
import os
import pickle
import time

from research.bohb.bohb_runner import (
    run_full_bohb,
    run_portfolio_bohb,
)

from research.bohb.tabpfn_portfolio import (
    run_tabpfn_portfolio,
)

from research.bohb.plot import (
    plot_comparison,
)


def main(args):

    os.makedirs("output", exist_ok=True)

    # Global wall-clock start
    experiment_start = time.time()

    # =================================================
    # Full BOHB
    # =================================================

    print("\n====================")
    print("Running Full BOHB")
    print("====================\n")

    full_results = run_full_bohb(
        dataset_name=args.dataset,
        data_root=args.data_root,
        n_iterations=args.iterations,
        seed=args.seed,
    )

    # =================================================
    # Portfolio generation
    # =================================================

    print("\n====================")
    print("Generating Portfolio")
    print("====================\n")

    portfolio_configs = run_tabpfn_portfolio(
        dataset_name=args.dataset,
        data_root=args.data_root,
        target="accuracy",
        k=args.portfolio_k,
    )

    print(f"Selected portfolio size: {len(portfolio_configs)}")

    # =================================================
    # Portfolio BOHB
    # =================================================

    print("\n====================")
    print("Running Portfolio BOHB")
    print("====================\n")

    portfolio_iterations = max(
        1,
        args.iterations // args.portfolio_k,
    )

    portfolio_results = run_portfolio_bohb(
        portfolio_configs=portfolio_configs,
        dataset_name=args.dataset,
        data_root=args.data_root,
        n_iterations=portfolio_iterations,
        seed=args.seed,
    )

    # =================================================
    # Save raw results
    # =================================================

    result_file = f"output/{args.dataset}_bohb_results.pkl"

    with open(result_file, "wb") as f:

        pickle.dump(
            {
                "full_results": full_results,
                "portfolio_results": portfolio_results,
                "portfolio_configs": portfolio_configs,
                "seed": args.seed,
            },
            f,
        )

    print(f"Saved raw results to {result_file}")

    # =================================================
    # (Optional) Save best configs
    # =================================================
    #
    # json_file = f"outputs/{args.dataset}_best_results.json"
    #
    # best_results = {
    #     "dataset": args.dataset,
    #     "seed": args.seed,
    #     "full_bohb": extract_best_run(full_results),
    #     "portfolio_bohb": extract_best_portfolio(portfolio_results),
    #     "portfolio_configs": portfolio_configs,
    # }
    #
    # with open(json_file, "w") as f:
    #     json.dump(best_results, f, indent=4)

    # =================================================
    # Plot
    # =================================================

    print("\nGenerating comparison plot")

    try:

        plot_comparison(
            full_results=full_results,
            portfolio_results=portfolio_results,
            experiment_start=experiment_start,
            save_path=f"output/{args.dataset}_comparison.png",
        )

        print(f"Saved plot to output/{args.dataset}_comparison.png")

    except Exception as e:

        print(f"Plot generation failed: {e}")

    print("\nFinished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Full BOHB vs Portfolio BOHB")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--portfolio_k",
        type=int,
        default=1,
        choices=[1, 2, 3],
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of BOHB iterations",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    main(args)
