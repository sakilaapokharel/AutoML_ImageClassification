class Results:

    def __init__(self):
        self.results = []


    def add(
        self,
        config,
        score,
    ):

        self.results.append(
            {
                "config": config,
                "score": score,
            }
        )


    def best(self):

        return max(
            self.results,
            key=lambda x: x["score"],
        )


    def all(self):

        return self.results