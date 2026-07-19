class Evaluator:

    def evaluate(
        self,
        dataset_class,
        config,
        seed,
    ):
        raise NotImplementedError
    

class VisionEvaluator(Evaluator):

    def evaluate(
        self,
        dataset_class,
        config,
        seed,
    ):

        # create transform
        # sample fidelity
        # create train/val split
        # get encoder
        # create embeddings
        # train TabPFN
        # return validation accuracy

        # return score
        pass