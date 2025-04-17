from improvelib.utils import str2bool

preprocess_params = [
    {"name": "cutoff",
     "type": float,
     "default": 9.24,
     "help": "Cutoff for binarization. 10 was used in the paper, 9.24 is used for updated DrugComb v1.5 to allow for the same number of positive samples. ",
    },
]

train_params = []

infer_params = []