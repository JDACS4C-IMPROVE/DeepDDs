# IMPROVE - DeepDDS: Drug Synergy Prediction

---

This is the IMPROVE implementation of the original model with original data.

## Dependencies and Installation
### Conda Environment
```
conda create -n deepdds python pytorch-gpu scikit-learn pandas pytorch_geometric seaborn rdkit pyyaml
conda activate deepdds
```

### Clone this repository
```
git clone https://github.com/JDACS4C-IMPROVE/DeepDDs
cd DeepDDs
git checkout IMPROVE-original
cd ..
```

### Clone IMPROVE repository v0.1.0
```
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE
git checkout develop
cd ..
```

### Download Original Data
Data is provided in the repo, under 'data'.


## Running the Model
Activate the conda environment:

```
conda activate deepdds
```

Set environment variables:
```
export PYTHONPATH=$PYTHONPATH:/your/path/to/IMPROVE
```

Run preprocess, train, and infer scripts:
```
cd DeepDDs
python deepdds_preprocess_improve.py
python deepdds_train_improve.py
python deepdds_infer_improve.py
```



## References

