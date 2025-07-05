""" Preprocess data to generate datasets for the prediction model.
"""

import sys
from pathlib import Path
from typing import Dict
# [Req] Core improvelib imports
from improvelib.applications.synergy.config import SynergyPreprocessConfig
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
    # [Req] Load feature data
    # ------------------------------------------------------
    print("Load omics data.")
    cell_feature = frm.get_x_data(file = params['cell_transcriptomic_file'], 
                                        benchmark_dir = params['input_dir'], 
                                        column_name = params['canc_col_name'])

    print("Load drug data.")
    drug_feature = frm.get_x_data(file = params['drug_smiles_file'], 
                    benchmark_dir = params['input_dir'], 
                    column_name = params['drug_col_name'])

    # ------------------------------------------------------
    # [Req] Validity check of feature representations
    # ------------------------------------------------------
    smi_to_drop = []
    drug_feature.columns = ['SMILES']
    for i, row in drug_feature.iterrows():
        try:
            smile_to_graph(row['SMILES'])
        except:
            print(f"Invalid SMILE string {row['SMILES']}, ID is {i}, removing from analysis.")
            smi_to_drop = smi_to_drop + [i]   
    drug_feature = drug_feature.drop(smi_to_drop)


    # ------------------------------------------------------
    # [Req] Determine preprocessing on training data
    # ------------------------------------------------------
    print("Load train response data.")
    response_train = frm.get_y_data(split_file=params["train_split_file"], 
                                   benchmark_dir=params['input_dir'], 
                                   y_data_file=params['y_data_file'])
    response_train = response_train.dropna(subset=[params['y_col_name']])
    
    print("Find intersection of training data.")
    response_train = frm.get_y_data_with_features(response_train, cell_feature, params['canc_col_name'])
    response_train = frm.get_y_data_with_features(response_train, drug_feature, params['drug_1_col_name'])
    response_train = frm.get_y_data_with_features(response_train, drug_feature, params['drug_2_col_name'])
    omics_train = frm.get_features_in_y_data(cell_feature, response_train, params['canc_col_name'])

    print("Determine transformations.")
    frm.determine_transform(omics_train, 'omics_transform', params['cell_transcriptomic_transform'], params['output_dir'])



    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------
    # Dict with split files corresponding to the three sets (train, val, and test)
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():
        print(f"Prepare data for stage {stage}.")
        print(f"Find intersection of {stage} data.")
        y_data_stage = frm.get_y_data(split_file=split_file, 
                                benchmark_dir=params['input_dir'], 
                                y_data_file=params['y_data_file'])
        y_data_stage = y_data_stage.dropna(subset=[params['y_col_name']])
        y_data_stage = frm.get_y_data_with_features(y_data_stage, cell_feature, params['canc_col_name'])
        y_data_stage = frm.get_y_data_with_features(y_data_stage, drug_feature, params['drug_1_col_name'])
        y_data_stage = frm.get_y_data_with_features(y_data_stage, drug_feature, params['drug_2_col_name'])
        omics_stage = frm.get_features_in_y_data(cell_feature, y_data_stage, params['canc_col_name'])
        drug1_stage = frm.get_features_in_y_data(drug_feature, y_data_stage, params['drug_1_col_name'])
        drug2_stage = frm.get_features_in_y_data(drug_feature, y_data_stage, params['drug_2_col_name'])
        drugs_stage = pd.concat([drug1_stage, drug2_stage]).drop_duplicates()

        print(f"Transform {stage} data.")
        omics_stage = frm.transform_data(omics_stage, 'omics_transform', params['output_dir'])

        # binarize the y data
        synergy_bins = [-np.inf, params['cutoff'], np.inf]
        synergy_labels = [0, 1]
        y_data_stage['label'] = pd.cut(np.array(y_data_stage[params['y_col_name']]), bins=synergy_bins, labels=synergy_labels)
        # merge y_data with drug data
        data = y_data_stage.merge(drugs_stage, how='inner', left_on=params['drug_1_col_name'], right_on='DrugID')
        data = data.rename(columns={'SMILES': 'drug1'})
        data = data.merge(drugs_stage, how='inner', left_on=params['drug_2_col_name'], right_on='DrugID')
        data = data.rename(columns={'SMILES': 'drug2'})
        data = data.rename(columns={params['canc_col_name']: 'cell'})
    
        cell_feature_for_TBD = np.array(omics_stage.reset_index())
        cell_feature_for_TBD = cell_feature_for_TBD.astype(str)

        smile_graph = {}
        for i, row in drug_feature.iterrows():
            g = smile_to_graph(row['SMILES'])
            smile_graph[row['SMILES']] = g


        drug1_stage, drug2_stage, cell_stage, label_stage = np.asarray(list(data['drug1'])), np.asarray(list(data['drug2'])), np.asarray(list(data['cell'])), np.asarray(list(data['label']))
        print('开始创建数据 - Start creating data')
        drug1_data_train = TestbedDataset(root=params['output_dir'], dataset=f"drug1_{stage}", xd=drug1_stage, xt=cell_stage, xt_featrue=cell_feature_for_TBD, y=label_stage, smile_graph=smile_graph)
        drug2_data_train = TestbedDataset(root=params['output_dir'], dataset=f"drug2_{stage}", xd=drug2_stage, xt=cell_stage, xt_featrue=cell_feature_for_TBD, y=label_stage, smile_graph=smile_graph)
        print('创建数据成功 - Data created successfully')
        frm.save_stage_ydf(ydf=y_data_stage, stage=stage, output_dir=params["output_dir"])

    return params["output_dir"]


# [Req]
def main(args):
    cfg = SynergyPreprocessConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="deepdds_params.ini",
                                       additional_definitions=preprocess_params)
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])