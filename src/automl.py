from automl.search import Search
from automl.evaluator import Evaluator
from automl.results import Results


class AutoML:

    def __init__(
        self,
        search: Search,
        evaluator: Evaluator,
        seed: int = 42,
    ):
        self.search = search
        self.evaluator = evaluator
        self.seed = seed

        self.results = Results()


    def fit(self, dataset_class):

        for config in self.search:

            score = self.evaluator.evaluate(
                dataset_class,
                config,
                seed=self.seed,
            )

            self.results.add(
                config,
                score,
            )


        return self


    def get_best_config(self):

        return self.results.best()



    def get_results(self):

        return self.results.all()