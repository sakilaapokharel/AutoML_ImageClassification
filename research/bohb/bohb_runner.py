# bohb/bohb_runner.py


import random
import numpy as np
import torch


from hpbandster.optimizers import BOHB
from hpbandster.core.nameserver import NameServer


from research.bohb.config import (
    get_full_search_space,
    get_hyperparameter_search_space,
)


from research.bohb.bohb_worker import CNNWorker

# =====================================================
# Reproducibility
# =====================================================


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =====================================================
# Full BOHB
# =====================================================


def run_full_bohb(
    dataset_name,
    data_root,
    n_iterations=3,
    min_budget=1,
    max_budget=9,
    seed=0,
):

    set_seed(seed)

    cs = get_full_search_space()

    run_id = "full_bohb"

    # -----------------------------
    # NameServer
    # -----------------------------

    NS = NameServer(
        run_id=run_id,
        host="127.0.0.1",
        port=0,
    )

    ns_host, ns_port = NS.start()

    # -----------------------------
    # Worker
    # -----------------------------

    worker = CNNWorker(
        nameserver=ns_host,
        nameserver_port=ns_port,
        run_id=run_id,
        dataset_name=dataset_name,
        data_root=data_root,
    )

    worker.run(background=True)

    # -----------------------------
    # BOHB
    # -----------------------------

    bohb = BOHB(
        configspace=cs,
        run_id=run_id,
        nameserver=ns_host,
        nameserver_port=ns_port,
        min_budget=min_budget,
        max_budget=max_budget,
    )

    try:

        results = bohb.run(n_iterations)

    finally:

        bohb.shutdown(shutdown_workers=True)

        NS.shutdown()

    return results


# =====================================================
# Portfolio BOHB
# =====================================================


def run_portfolio_bohb(
    portfolio_configs,
    dataset_name,
    data_root,
    n_iterations=3,
    min_budget=1,
    max_budget=9,
    seed=0,
):

    set_seed(seed)

    cs = get_hyperparameter_search_space()

    all_results = []

    for idx, portfolio_config in enumerate(portfolio_configs):

        run_id = f"portfolio_bohb_{idx}"

        print(f"\nPortfolio {idx+1}/{len(portfolio_configs)}")

        # -----------------------------
        # NameServer
        # -----------------------------

        NS = NameServer(
            run_id=run_id,
            host="127.0.0.1",
            port=0,
        )

        ns_host, ns_port = NS.start()

        # -----------------------------
        # Worker
        # -----------------------------

        worker = CNNWorker(
            nameserver=ns_host,
            nameserver_port=ns_port,
            run_id=run_id,
            dataset_name=dataset_name,
            data_root=data_root,
            fixed_config=portfolio_config,
        )

        worker.run(background=True)

        # -----------------------------
        # BOHB
        # -----------------------------

        bohb = BOHB(
            configspace=cs,
            run_id=run_id,
            nameserver=ns_host,
            nameserver_port=ns_port,
            min_budget=min_budget,
            max_budget=max_budget,
        )

        try:

            results = bohb.run(n_iterations)

        finally:

            bohb.shutdown(shutdown_workers=True)

            NS.shutdown()

        all_results.append(
            {
                "portfolio_config": portfolio_config,
                "results": results,
            }
        )

    return all_results
