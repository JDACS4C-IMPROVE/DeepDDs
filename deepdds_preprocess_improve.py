""" Preprocess data to generate datasets for the prediction model.
"""

import sys
from pathlib import Path
from typing import Dict
# [Req] Core improvelib imports
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
from improvelib.utils import str2bool
import improvelib.utils as frm

# Model-specific imports
import csv
from itertools import islice
import joblib
import pandas as pd
import numpy as np
#import os
#import json, pickle
from collections import OrderedDict
from rdkit import Chem
# from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils_test import TestbedDataset
import random
from random import shuffle
import torch.utils.data as Data
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset
import torch_geometric.deprecation

from model_params_def import preprocess_params
filepath = Path(__file__).resolve().parent # [Req]



def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1: ]

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


# [Req]
def run(params: Dict):


    # ------------------------------------------------------
    # Load X data (feature representations)
    # ------------------------------------------------------
    file2 = 'data/independent_set/independent_cell_features_954.csv'
    #file2 = cellfile
    cell_features = []
    with open(file2) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:
            cell_features.append(row)
    cell_features = np.array(cell_features)
    print('cell_features', cell_features)

    compound_iso_smiles = []
    df = pd.read_csv('data/smiles.csv')
    compound_iso_smiles += list(df['smile'])
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    print('compound_iso_smiles', compound_iso_smiles)
    for smile in compound_iso_smiles:
        print('smiles', smile)
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    # ------------------------------------------------------
    # Load Y data 
    # ------------------------------------------------------
    y_data = 'new_labels_0_10'
    # convert to PyTorch data format

    df = pd.read_csv('data/' + y_data + '.csv')





    # ------------------------------------------------------
    # Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------

    lenth = len(df)
    pot = int(lenth/5)
    print('lenth', lenth)
    print('pot', pot)

    random_num = random.sample(range(0, lenth), lenth)
    #for i in range(1):
    i=0
    test_num = random_num[pot*i:pot*(i+1)]
    train_num = random_num[:pot*i] + random_num[pot*(i+1):]

    df_train = df.iloc[train_num]
    df_test = df.iloc[test_num]

    drug1_train, drug2_train, cell_train, label_train = list(df_train['drug1']), list(df_train['drug2']), list(df_train['cell']), list(df_train['label'])
    drug1_train, drug2_train, cell_train, label_train = np.asarray(drug1_train), np.asarray(drug2_train), np.asarray(cell_train), np.asarray(label_train)
    # make data PyTorch Geometric ready

    print("TRAIN")
    print('开始创建数据 - Start creating data')
    drug1_data_train = TestbedDataset(root=params['output_dir'], dataset='drug1_train', xd=drug1_train, xt=cell_train, xt_featrue=cell_features, y=label_train,smile_graph=smile_graph)
    drug2_data_train = TestbedDataset(root=params['output_dir'], dataset='drug2_train', xd=drug2_train, xt=cell_train, xt_featrue=cell_features, y=label_train,smile_graph=smile_graph)
    print('创建数据成功 - Data created successfully')

    drug1_test, drug2_test, cell_test, label_test = list(df_test['drug1']), list(df_test['drug2']), list(df_test['cell']), list(df_test['label'])
    drug1_test, drug2_test, cell_test, label_test = np.asarray(drug1_test), np.asarray(drug2_test), np.asarray(cell_test), np.asarray(label_test)
    # make data PyTorch Geometric ready

    print("TEST")
    print('开始创建数据 - Start creating data')
    drug1_data_test = TestbedDataset(root=params['output_dir'], dataset='drug1_test', xd=drug1_test, xt=cell_test, xt_featrue=cell_features, y=label_test,smile_graph=smile_graph)
    drug2_data_test = TestbedDataset(root=params['output_dir'], dataset='drug2_test', xd=drug2_test, xt=cell_test, xt_featrue=cell_features, y=label_test,smile_graph=smile_graph)
    print('创建数据成功 - Data created successfully')

    return params["output_dir"]


# [Req]
def main(args):
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="deepdds_params.txt",
        additional_definitions=preprocess_params)
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])