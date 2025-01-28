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
import random
import numpy as np
import pandas as pd
from random import shuffle
import torch.utils.data as Data
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset
from models.gat import GATNet
from models.gat_gcn_test import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils_test import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold



filepath = Path(__file__).resolve().parent # [Req]


# ---------------------
modeling = GCNNet
#TRAIN_BATCH_SIZE = 256
#TEST_BATCH_SIZE = 256
#LR = 0.0005
#LOG_INTERVAL = 20
#NUM_EPOCHS = 1000

# training function at each epoch
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    # train_loader = np.array(train_loader)
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


    print('Learning rate: ', params["learning_rate"])
    print('Epochs: ', params["epochs"])
    datafile = 'new_labels_0_10'



    # --------------------------------------------------------------------
    # [Req] Create data names for train/val sets and build model path
    # --------------------------------------------------------------------
    train_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")  # [Req]
    val_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")  # [Req]

    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"])

    train_data_path = params["input_dir"] + "/" + train_data_fname
    val_data_path = params["input_dir"] + "/" + val_data_fname


    
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

    drug1_train_path = params["input_dir"] + "/" + "drug1_train.pt"
    drug1_test_path = params["input_dir"] + "/" + "drug1_test.pt"
    drug2_train_path = params["input_dir"] + "/" + "drug2_train.pt"
    drug2_test_path = params["input_dir"] + "/" + "drug2_test.pt"
    drug1_data_train = torch.load(drug1_train_path)
    drug1_data_test = torch.load(drug1_test_path)
    drug2_data_train = torch.load(drug2_train_path)
    drug2_data_test = torch.load(drug2_test_path)

    print("torch load")

    #drug1_data_train = drug1_data[train_num]
    #drug1_data_test = drug1_data[test_num]
    drug1_loader_train = DataLoader(drug1_data_train, batch_size=params["batch_size"], shuffle=None)
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=params["batch_size"], shuffle=None)
    #drug2_data_test = drug2_data[test_num]
    #drug2_data_train = drug2_data[train_num]
    drug2_loader_train = DataLoader(drug2_data_train, batch_size=params["batch_size"], shuffle=None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=params["batch_size"], shuffle=None)

    print("data load")
  
    # ------------------------------------------------------
    # Prepare model
    # ------------------------------------------------------
    model = modeling().to(device)
    global loss_fn
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])


    model_file_name = 'data/result/GCNNet(DrugA_DrugB)' + '--model_' + datafile +  '.model'
    result_file_name = 'data/result/GCNNet(DrugA_DrugB)' + '--result_' + datafile +  '.csv'
    file_AUCs = 'data/result/GCNNet(DrugA_DrugB)' + '--AUCs--' + datafile + '.txt'
    AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')
    # -----------------------------
    # Train. Iterate over epochs.
    # -----------------------------

    best_auc = 0
    for epoch in range(params["epochs"]):
        train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)
        T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)
        # T is correct label
        # S is predict score
        # Y is predict label

        # compute preformence
        AUC = roc_auc_score(T, S)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
        PREC = precision_score(T, Y)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        recall = recall_score(T, Y)

        # save data
        AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall]
        #save_AUCs(AUCs, file_AUCs)
        ret = [rmse(T, S), mse(T, S), pearson(T, S), spearman(T, S), ci(T, S)]
        if best_auc < AUC:
            best_auc = AUC
            print(best_auc)
            save_AUCs(AUCs, file_AUCs)
    # -----------------------------
    # Save model
    # -----------------------------
    torch.save(model, modelpath)
    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    best_model = torch.load(modelpath, weights_only=False)
    T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)
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
        output_dir=params["output_dir"]
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