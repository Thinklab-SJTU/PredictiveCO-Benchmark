from openpto.problems.PTOProblem import PTOProblem


class Advertising(PTOProblem):
    """ """

    def __init__(self, data_dir="./openpto/data/", **kwargs):
        super(Advertising, self).__init__()
        self.data_dir = data_dir

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

    def get_twostageloss(self):
        raise NotImplementedError()

    def get_decision(self, Y, params, optSolver=None, isTrain=True, **kwargs):
        raise NotImplementedError()

    def get_objective(self, Y, Z, **kwargs):
        raise NotImplementedError()
