import torchvision

from openpto.method.Predicts.cv_model import CombResnet18, PureConvNet, cv_mlp
from openpto.method.Predicts.cvr_model import CVRModel
from openpto.method.Predicts.dense import MLP


######################## prediction model wrapper  ############################
def pred_model_wrapper(args, pred_model_args):
    model_dict = {
        "dense": MLP,
        "cvr": CVRModel,
        "cv_mlp": cv_mlp,
        "resnet18": torchvision.models.resnet18,
        "CombResnet18": CombResnet18,
        "PureConvNet": PureConvNet,
    }
    return model_dict[args.pred_model](
        num_features=pred_model_args["ipdim"],
        num_targets=pred_model_args["opdim"],
        num_layers=args.n_layers,
        intermediate_size=args.n_hidden,
        output_activation=pred_model_args["out_act"],
    )
