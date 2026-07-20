# bohb_worker.py

import time
import numpy as np

from hpbandster.core.worker import Worker

from research.bohb.train import train_model


class CNNWorker(Worker):

    def __init__(
        self, *args, dataset_name=None, data_root=None, fixed_config=None, **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.dataset_name = dataset_name

        self.data_root = data_root

        self.fixed_config = fixed_config

    # -----------------------------------------
    # Convert numpy types to Python types
    # for Pyro4 serialization
    # -----------------------------------------

    def clean_config(self, cfg):

        clean = {}

        for k, v in cfg.items():

            if isinstance(v, np.generic):

                clean[str(k)] = v.item()

            else:

                clean[str(k)] = v

        return clean

    def compute(self, config, budget, working_directory, **kwargs):

        start = time.time()

        # BOHB config
        config = self.clean_config(config)

        # Portfolio fixed architecture
        if self.fixed_config is not None:

            fixed_config = self.clean_config(self.fixed_config)

            final_config = {**fixed_config, **config}

        else:

            final_config = config

        result = train_model(
            config=final_config,
            epochs=int(budget),
            dataset_name=self.dataset_name,
            data_root=self.data_root,
        )

        accuracy = float(result["val_accuracy"])

        compute_time = float(time.time() - start)

        # IMPORTANT:
        # return only Python native objects
        return {
            "loss": float(1.0 - accuracy),
            "info": {
                "val_accuracy": accuracy,
                "compute_time_sec": compute_time,
                "config": final_config,
            },
        }
