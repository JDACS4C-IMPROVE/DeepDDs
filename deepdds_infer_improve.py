import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from model_params_def import infer_params

# Model-specific imports, as needed
from deepdds_train_improve import predicting
from utils_test import DataLoader
import torch

filepath = Path(__file__).resolve().parent # [Req]


# [Req]
def run(params):
    # --------------------------------------------------------------------
    # [Req] Create data names for test set and build model path
    # --------------------------------------------------------------------
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["input_model_dir"])

    # --------------------------------------------------------------------
    # Load inference data (ML data)
    # --------------------------------------------------------------------

    # original model only has train and test, so reusing test data for both val and test
    drug1_test_path = params["input_data_dir"] + "/" + "drug1_test.pt"
    drug2_test_path = params["input_data_dir"] + "/" + "drug2_test.pt"
    drug1_data_test = torch.load(drug1_test_path)
    drug2_data_test = torch.load(drug2_test_path)
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=params["infer_batch"], shuffle=None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=params["infer_batch"], shuffle=None)

    # --------------------------------------------------------------------
    # CUDA/CPU device, as needed
    # --------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    # --------------------------------------------------------------------
    # Load best model and compute predictions
    # --------------------------------------------------------------------

    best_model = torch.load(modelpath, weights_only=False)
    T, S, Y = predicting(best_model, device, drug1_loader_test, drug2_loader_test)
        # T is correct label
        # S is predict score
        # Y is predict label
    print("T:", T)
    print("S:", S)
    print("Y:", Y)

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=T,
        y_pred=Y,
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_data_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=T,
            y_pred=Y,
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )

    return True


# [Req]
def main(args):
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="deepdds_params.txt",
        additional_definitions=infer_params,
    )
    status = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])