from openpto.problems.PTOProblem import PTOProblem


class Shortestpath(PTOProblem):
    """The budget allocation predict-then-optimise problem from Wilder et. al. (2019)"""

    def __init__(
        self,
        num_train_instances=100,  # number of instances to use from the dataset to train
        num_test_instances=500,  # number of instances to use from the dataset to test
        num_items=100,  # number of targets to consider
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
    ):
        super(Shortestpath, self).__init__()
        self._set_seed(rand_seed)

    @staticmethod
    def gendata():
        return

    def get_train_data(self, **kwargs):
        raise NotImplementedError()

    def get_val_data(self, **kwargs):
        raise NotImplementedError()

    def get_test_data(self, **kwargs):
        raise NotImplementedError()

    def get_model_shape(self):
        raise NotImplementedError()

    def get_output_activation(self):
        raise NotImplementedError()
