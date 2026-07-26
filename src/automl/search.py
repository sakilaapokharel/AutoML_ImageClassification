from itertools import product


class Search:
    """
    Grid search over a configuration space.

    Example:
        config_space = {
            "encoder": [
                "levit_128s",
                "tinyvit_5m",
                "deit_tiny",
                "mobilenetv3_small",
                "edgenext_xx_small",
                "resnet18",
            ],
            "embedding_dim": [
                None,
                128,
                64,
            ]
        }

        search = Search(config_space)

        for config in search:
            print(config)
    """

    def __init__(self, config_space):
        self.config_space = config_space
        self.configs = self._build_configs()

    def _build_configs(self):
        if not self.config_space:
            return []

        keys = self.config_space.keys()
        values = self.config_space.values()

        return [dict(zip(keys, combination)) for combination in product(*values)]

    def __iter__(self):
        return iter(self.configs)

    def __len__(self):
        return len(self.configs)
