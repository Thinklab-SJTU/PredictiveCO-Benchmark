from openpto.method.Predicts.cvr_model import CVRModel
from openpto.method.Predicts.dense import MLP


######################## prediction model wrapper  ############################
def pred_model_wrapper(args, pred_model_args):
    model_dict = {"dense": MLP, "cvr": CVRModel}
    return model_dict[args.pred_model](
        num_features=pred_model_args["ipdim"],
        num_targets=pred_model_args["opdim"],
        num_layers=args.n_layers,
        intermediate_size=args.n_hidden,
        output_activation=pred_model_args["out_act"],
    )
