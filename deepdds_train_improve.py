""" Train model for synergy prediction.
"""

import sys
from pathlib import Path
from typing import Dict

# [Req] IMPROVE imports
from improvelib.applications.synergy.config import SynergyTrainConfig
#from improvelib.utils import str2bool
import improvelib.utils as frm
#from improvelib.metrics import compute_metrics
from model_params_def import train_params

# Model-specific imports
import numpy as np
import torch
import torch.nn.functional as F
#import torch.utils.data as Data
import torch.nn as nn
#from torch.utils.data import TensorDataset, Dataset
from torch_geometric.data import DataLoader
from models.gat import GATNet
from models.gat_gcn_test import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils_test import TestbedDataset, train, predicting, determine_sample_data
from sklearn.metrics import roc_auc_score

filepath = Path(__file__).resolve().parent # [Req]
modeling = GCNNet

def run(params):
    # --------------------------------------------------------------------
    # [Req] Create data names for train/val sets and build model path
    # --------------------------------------------------------------------
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"])

    # ------------------------------------------------------
    # CUDA/CPU device
    # ------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------
    drug1_data_train = TestbedDataset(root=params['input_dir'], dataset='drug1_train')
    drug2_data_train = TestbedDataset(root=params['input_dir'], dataset='drug2_train')
    drug1_data_val = TestbedDataset(root=params['input_dir'], dataset='drug1_val')
    drug2_data_val = TestbedDataset(root=params['input_dir'], dataset='drug2_val')
    print("torch load")

    drug1_loader_train = DataLoader(drug1_data_train, batch_size=params["batch_size"], shuffle=None)
    drug2_loader_train = DataLoader(drug2_data_train, batch_size=params["batch_size"], shuffle=None)
    drug1_loader_val = DataLoader(drug1_data_val, batch_size=params["val_batch"], shuffle=None)
    drug2_loader_val = DataLoader(drug2_data_val, batch_size=params["val_batch"], shuffle=None)
    print("data load")
  
    # ------------------------------------------------------
    # Prepare model
    # ------------------------------------------------------
    determine_sample_data(drug1_loader_train)
    model = modeling(num_features_xt=drug1_data_train.cell.shape[1]).to(device)
    global loss_fn
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    # -----------------------------
    # Train. Iterate over epochs.
    # -----------------------------
    best_auc = 0
    early_stop = 0
    for epoch in range(params["epochs"]):
        if early_stop < params["patience"]:
            train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1, loss_fn)
            T, S, Y = predicting(model, device, drug1_loader_val, drug2_loader_val)
            AUC = roc_auc_score(T, S)
            early_stop = early_stop + 1
            if best_auc < AUC:
                best_auc = AUC
                print("AUC improved to:", best_auc)
                print("Saving model.")
                torch.save(model, modelpath)
                early_stop = 0
            print(early_stop, "epochs since last improvement out of", params['patience'], "for early stopping.")

    
    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    best_model = torch.load(modelpath, weights_only=False)
    T, S, Y = predicting(best_model, device, drug1_loader_val, drug2_loader_val)
        # T is correct label
        # S is predict score
        # Y is predict label


    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=T,
        y_pred=Y,
        stage="val",
        y_col_name='label',
        output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )


    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performance_scores(
        y_true=T,
        y_pred=Y,
        stage="val",
        metric_type=params["metric_type"],
        output_dir=params["output_dir"],
        y_prob=S
    )

    return val_scores


def main(args):
    cfg = SynergyTrainConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="deepdds_params.ini",
        additional_definitions=train_params)
    val_scores = run(params)
    print("\nFinished training model.")



# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])