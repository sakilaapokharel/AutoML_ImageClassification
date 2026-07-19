from itertools import product


class Search:

    def __init__(
        self,
        config_space,
        fidelity,
        dataset_cls=None,
    ):
        self.config_space = config_space
        self.dataset_cls = dataset_cls
        self.fidelity = fidelity
        self.configs = self._build_configs()

    def _valid_config(
        self,
        config,
    ):

        if self.dataset_cls is None:
            return True

        width = self.dataset_cls.width
        height = self.dataset_cls.height
        # Do not use resize=None for small images
        if width < 32 or height < 32:
            if config.get("resize") == None:

                if config.get("encoder") == "densenet121":
                    return False

        if config.get("embedding_dim") is not None:

            if self.fidelity is not None:

                max_components = self.dataset_cls.num_classes * self.fidelity

                if config["embedding_dim"] > max_components:
                    return False

        return True

    def _build_configs(self):

        keys = list(self.config_space.keys())
        values = list(self.config_space.values())

        configs = []

        for combination in product(*values):

            config = dict(zip(keys, combination))

            if self._valid_config(config):
                configs.append(config)
        return configs

    def __iter__(self):

        yield from self.configs

    def size(self):

        return len(self.configs)
