""" Preprocess data to generate datasets for the prediction model.
"""

import sys
from pathlib import Path
from typing import Dict
# [Req] Core improvelib imports
from improvelib.applications.synergy.config import SynergyPreprocessConfig
import improvelib.applications.synergy.synergy_utils as syn
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
    y_data = syn.get_all_response_data(train_split_file = params['train_split_file'], 
                                   val_split_file = params['val_split_file'], 
                                   test_split_file = params['test_split_file'], 
                                   benchmark_dir = params['input_dir'])
    cell_feature = syn.get_cell_transcriptomics(file = params['cell_transcriptomic_file'], 
                                                  benchmark_dir = params['input_dir'], 
                                                  cell_column_name = params['cell_column_name'], 
                                                  norm = params['cell_transcriptomic_transform'])
    drug_feature = syn.get_drug_smiles(file = params['drug_smiles_file'], 
                     benchmark_dir = params['input_dir'], 
                     drug_column_name = params['drug_column_name'])
    
    #file2 = 'data/independent_set/independent_cell_features_954.csv'
    #cell_features = []
    #with open(file2) as csvfile:
    #    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    #    for row in csv_reader:
    #        cell_features.append(row)
    #cell_features = np.array(cell_features)
    #print('cell_features', cell_features)
    cell_features = np.array(cell_feature)

    #compound_iso_smiles = []
    #df = pd.read_csv('data/smiles.csv')
    drug_feature_cleaned = drug_feature.dropna(subset=[drug_feature.columns[0]])
    compound_iso_smiles = list(drug_feature_cleaned.iloc[:, 0])
    #compound_iso_smiles += list(df['smile'])
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
    #y_data = 'new_labels_0_10'
    # convert to PyTorch data format
    #df = pd.read_csv('data/' + y_data + '.csv')


    # cutoff is 10 per paper -- unhardcode this
    synergy_bins = [-np.inf, 10, np.inf]
    synergy_labels = [0, 1]
    y_data['label'] = pd.cut(np.array(y_data[params['y_col_name']]), bins=synergy_bins, labels=synergy_labels)
    
    y_data = y_data.merge(drug_feature_cleaned, how='inner', left_on='DrugID_row', right_on='DrugID')
    y_data = y_data.drop('DrugID', axis=1)
    y_data = y_data.rename(columns={'smiles': 'drug1'})
    y_data = y_data.merge(drug_feature_cleaned, how='inner', left_on='DrugID_col', right_on='DrugID')
    y_data = y_data.drop('DrugID', axis=1)
    y_data = y_data.rename(columns={'smiles': 'drug2'})
    y_data = y_data.rename(columns={'DepMapID': 'cell'})

    small_y_data = y_data[['drug1', 'drug2', 'cell', 'label']]
    # ------------------------------------------------------
    # Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------

    #lenth = len(df)
    #pot = int(lenth/5)
    #print('lenth', lenth)
    #print('pot', pot)
    #random_num = random.sample(range(0, lenth), lenth)
    #i=0
    #test_num = random_num[pot*i:pot*(i+1)]
    #train_num = random_num[:pot*i] + random_num[pot*(i+1):]
    #df_train = df.iloc[train_num]
    #df_test = df.iloc[test_num]

    df_train = small_y_data[small_y_data['split'] == 'train']
    df_val = small_y_data[small_y_data['split'] == 'val']
    df_test = small_y_data[small_y_data['split'] == 'test']

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
    drug1_data_test = TestbedDataset(root=params['output_dir'], dataset='drug1_test', xd=drug1_test, xt=cell_test, xt_featrue=cell_features, y=label_test, smile_graph=smile_graph)
    drug2_data_test = TestbedDataset(root=params['output_dir'], dataset='drug2_test', xd=drug2_test, xt=cell_test, xt_featrue=cell_features, y=label_test, smile_graph=smile_graph)
    print('创建数据成功 - Data created successfully')

    drug1_val, drug2_val, cell_val, label_val = list(df_val['drug1']), list(df_val['drug2']), list(df_val['cell']), list(df_val['label'])
    drug1_val, drug2_val, cell_val, label_val = np.asarray(drug1_val), np.asarray(drug2_val), np.asarray(cell_val), np.asarray(label_val)
    # make data PyTorch Geometric ready

    print("VAL")
    print('开始创建数据 - Start creating data')
    drug1_data_val = TestbedDataset(root=params['output_dir'], dataset='drug1_val', xd=drug1_val, xt=cell_val, xt_featrue=cell_features, y=label_val, smile_graph=smile_graph)
    drug2_data_val = TestbedDataset(root=params['output_dir'], dataset='drug2_val', xd=drug2_val, xt=cell_val, xt_featrue=cell_features, y=label_val, smile_graph=smile_graph)
    print('创建数据成功 - Data created successfully')

    return params["output_dir"]


# [Req]
def main(args):
    cfg = SynergyPreprocessConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="deepdds_params.ini",
        additional_definitions=preprocess_params)
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])