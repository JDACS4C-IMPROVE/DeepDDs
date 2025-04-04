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
import pandas as pd
import numpy as np
from utils_test import TestbedDataset
from random import shuffle
import torch.nn.functional as F
import torch.nn as nn
from utils_preprocess import smile_to_graph
from model_params_def import preprocess_params
filepath = Path(__file__).resolve().parent # [Req]

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
    

    cell_features = np.array(cell_feature.reset_index())
    cell_feature = cell_feature.astype(str)
    drug_feature_cleaned = drug_feature.dropna(subset=[drug_feature.columns[0]])
    compound_iso_smiles = list(drug_feature_cleaned.iloc[:, 0])
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    print('compound_iso_smiles', compound_iso_smiles)
    for smile in compound_iso_smiles:
        try:
            g = smile_to_graph(smile)
            smile_graph[smile] = g
        except:
            print(smile, "is invalid")
    print("cleaned smiles", smile_graph.keys())
    drug_feature_final = drug_feature_cleaned[drug_feature_cleaned[drug_feature_cleaned.columns[0]].isin(list(smile_graph.keys()))]
    # ------------------------------------------------------
    # Load Y data 
    # ------------------------------------------------------



    # cutoff is 10 per paper -- unhardcode this
    synergy_bins = [-np.inf, params['cutoff'], np.inf]
    synergy_labels = [0, 1]
    y_data['label'] = pd.cut(np.array(y_data[params['y_col_name']]), bins=synergy_bins, labels=synergy_labels)
    
    y_data = y_data.merge(drug_feature_final, how='inner', left_on='DrugID_row', right_on='DrugID')
    #y_data = y_data.drop('DrugID', axis=1)
    y_data = y_data.rename(columns={'smiles': 'drug1'})
    y_data = y_data.merge(drug_feature_final, how='inner', left_on='DrugID_col', right_on='DrugID')
    #y_data = y_data.drop('DrugID', axis=1)
    y_data = y_data.rename(columns={'smiles': 'drug2'})
    y_data = y_data.rename(columns={params['cell_column_name']: 'cell'})
    y_data = y_data[y_data['cell'].isin(cell_feature.index.to_list())]
    small_y_data = y_data[['drug1', 'drug2', 'cell', 'label', 'split']]
    # ------------------------------------------------------
    # Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------

    df_train = small_y_data[small_y_data['split'] == 'train']
    df_val = small_y_data[small_y_data['split'] == 'val']
    df_test = small_y_data[small_y_data['split'] == 'test']

    drug1_train, drug2_train, cell_train, label_train = list(df_train['drug1']), list(df_train['drug2']), list(df_train['cell']), list(df_train['label'])
    drug1_train, drug2_train, cell_train, label_train = np.asarray(drug1_train), np.asarray(drug2_train), np.asarray(cell_train), np.asarray(label_train)


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