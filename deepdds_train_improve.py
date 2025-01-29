""" Train model for synergy prediction.
"""

import sys
from pathlib import Path
from typing import Dict

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics
from model_params_def import train_params

# Model-specific imports
import numpy as np
from random import shuffle
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset
from models.gat import GATNet
from models.gat_gcn_test import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils_test import TestbedDataset
from sklearn.metrics import roc_auc_score

filepath = Path(__file__).resolve().parent # [Req]


# ---------------------
modeling = GCNNet

def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data[0].y.view(-1, 1).long().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = loss_fn(output, y)
        # print('loss', loss)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.x),
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))


def predicting(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output = model(data1, data2)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

# [Req]
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
    drug1_data_test = TestbedDataset(root=params['input_dir'], dataset='drug1_test')
    drug2_data_test = TestbedDataset(root=params['input_dir'], dataset='drug2_test')
    
    print("torch load")

    drug1_loader_train = DataLoader(drug1_data_train, batch_size=params["batch_size"], shuffle=None)
    drug2_loader_train = DataLoader(drug2_data_train, batch_size=params["batch_size"], shuffle=None)
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=params["val_batch"], shuffle=None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=params["val_batch"], shuffle=None)

    print("data load")
  
    # ------------------------------------------------------
    # Prepare model
    # ------------------------------------------------------
    model = modeling().to(device)
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
            train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)
            T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)
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
    T, S, Y = predicting(best_model, device, drug1_loader_test, drug2_loader_test)
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
        y_col_name=params["y_col_name"],
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




# [Req]
def main(args):
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="deepdds_params.txt",
        additional_definitions=train_params)
    val_scores = run(params)
    print("\nFinished training model.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])