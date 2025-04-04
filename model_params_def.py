from improvelib.utils import str2bool

preprocess_params = [
    {"name": "cutoff",
     "type": int,
     "default": 10,
     "help": "Cutoff for binarization. Default is 10 as per paper.",
    },
]

train_params = []

infer_params = []