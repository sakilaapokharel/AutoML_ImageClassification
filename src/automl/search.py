from itertools import product


class Search:

    def __init__(
        self,
        config_space,
    ):
        self.config_space = config_space

    def __iter__(self):

        keys = list(self.config_space.keys())
        values = list(self.config_space.values())

        for combination in product(*values):
            yield dict(zip(keys, combination))

    def size(self):

        result = 1

        for values in self.config_space.values():
            result *= len(values)

        return result
