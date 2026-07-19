from itertools import product


class Search:

    def __init__(
        self,
        space,
    ):
        self.space = space


    def __iter__(self):

        keys = self.space.keys()

        values = self.space.values()


        for combination in product(*values):

            yield dict(
                zip(keys, combination)
            )