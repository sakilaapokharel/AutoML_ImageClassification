import time


class SuccessiveHalvingSearch:

    def __init__(
        self,
        search,
        fidelities,
        reduction_factor,
    ):
        self.search = search
        self.fidelities = fidelities
        self.reduction_factor = reduction_factor

    def run(
        self,
        evaluate,
        save_stage,
    ):

        configs = list(self.search)

        for fidelity in self.fidelities:

            print("\n" + "=" * 80)
            print("Successive Halving Stage")
            print("=" * 80)

            print(f"Fidelity               : {fidelity}")

            print(f"Incoming configurations: {len(configs)}")

            print("=" * 80)

            stage_results = []

            for idx, config in enumerate(
                configs,
                start=1,
            ):

                print("\n" + "-" * 70)
                print(f"Running configuration {idx}/{len(configs)}")

                print(config)

                print("-" * 70)

                start = time.time()

                score = evaluate(
                    config,
                    fidelity,
                )

                compute_time = time.time() - start

                stage_results.append(
                    (
                        score,
                        config,
                        compute_time,
                    )
                )

                print(f"Finished configuration {idx}")

                print(f"Accuracy : {score:.4f}")

                print(f"Time     : {compute_time:.2f}s")

            # -----------------------------
            # Ranking
            # -----------------------------

            stage_results.sort(
                key=lambda x: x[0],
                reverse=True,
            )

            print("\n" + "=" * 80)
            print("Stage Ranking")
            print("=" * 80)

            for rank, (score, config, _) in enumerate(
                stage_results,
                start=1,
            ):

                print(f"{rank}. Score={score:.4f}")

                print(config)
                print()

            # -----------------------------
            # Halving
            # -----------------------------

            keep = max(
                1,
                len(stage_results) // self.reduction_factor,
            )

            survivors = stage_results[:keep]

            configs = [config for _, config, _ in survivors]

            print("\n" + "=" * 80)
            print("Successive Halving Selection")
            print("=" * 80)

            print(f"Evaluated configurations : {len(stage_results)}")

            print(f"Keeping configurations  : {keep}")

            for rank, (score, config, _) in enumerate(
                survivors,
                start=1,
            ):

                print("\n" + "-" * 50)

                print(f"Rank {rank}")

                print(f"Score: {score:.4f}")

                print(config)

            save_stage(fidelity)

            print("\nStage completed")

            if len(configs) == 1:

                print("Only one configuration remains.")

                break

        return configs
