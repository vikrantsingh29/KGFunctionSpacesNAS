#!/usr/bin/env python3
import json
from dicee.executer import Execute
import pytorch_lightning as pl
import argparse
import nni


def get_default_arguments(description=None , custom_args=None):
    """ Extends pytorch_lightning Trainer's arguments with ours """
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser(add_help=False))
    # Default Trainer param https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#methods
    # Knowledge graph related arguments
    parser.add_argument("--dataset_dir", type=str, default="KGs/KINSHIP",
                        help="The path of a folder containing train.txt, and/or valid.txt and/or test.txt"
                             ",e.g., KGs/UMLS")
    parser.add_argument("--sparql_endpoint", type=str, default=None,
                        help="An endpoint of a triple store, e.g. 'http://localhost:3030/mutagenesis/'. ")
    parser.add_argument("--path_single_kg", type=str, default=None,
                        help="Path of a file corresponding to the input knowledge graph")
    # Saved files related arguments
    parser.add_argument("--scoring_func", type=str, default="compositional",
                        help="for function spaces"
                             ",e.g., compositional , vtp , trilinear")
    parser.add_argument("--loss_function_new", type=str, default="bce",
                        help="for function spaces"
                             ",e.g., bce , margin_loss , l2")
    parser.add_argument("--path_to_store_single_run", type=str, default=None,
                        help="A single directory created that contains related data about embeddings.")
    parser.add_argument("--storage_path", type=str, default='Experiments',
                        help="A directory named with time of execution under --storage_path "
                             "that contains related data about embeddings.")
    parser.add_argument("--save_embeddings_as_csv", action="store_true",
                        help="A flag for saving embeddings in csv file.")
    parser.add_argument("--backend", type=str, default="pandas",
                        choices=["pandas", "polars", "rdflib"],
                        help='Backend for loading, preprocessing, indexing input knowledge graph.')
    # Model related arguments
    parser.add_argument("--model", type=str,
                        default="Polynomial",
                        choices=["ComplEx","convolutionalFS","Polynomial","ComplexNumber","NAS",
                                 "FMult2", "Keci", "ConEx", "AConEx", "ConvQ", "AConvQ","ConvO",
                                 "AConvO", "QMult","OMult", "Shallom", "DistMult", "TransE",
                                 "Pykeen_MuRE", "Pykeen_QuatE", "Pykeen_DistMult", "Pykeen_BoxE",
                                 "Pykeen_CP","Pykeen_HolE", "Pykeen_ProjE", "Pykeen_RotatE",
                                 "Pykeen_TransE", "Pykeen_TransF", "Pykeen_TransH",
                                 "Pykeen_TransR", "Pykeen_TuckER", "Pykeen_ComplEx"],
                        help="Available knowledge graph embedding models. "
                             "To use other knowledge graph embedding models available in python, e.g.,"
                             "**Pykeen_BoxE** and add this into choices")
    parser.add_argument('--optim', type=str, default='Adam',
                        help='An optimizer',
                        choices=['Adam', 'SGD'])
    parser.add_argument('--embedding_dim', type=int, default=2,
                        help='Number of dimensions for an embedding vector. ')
    parser.add_argument("--num_epochs", type=int, default=100, help='Number of epochs for training. ')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Mini batch size. If None, automatic batch finder is applied')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument('--callbacks', type=json.loads,
                        default={},
                        help='{"PPE":{ "last_percent_to_consider": 10}}'
                             '"Perturb": {"level": "out", "ratio": 0.2, "method": "RN", "scaler": 0.3}')
    parser.add_argument("--trainer", type=str, default='torchCPUTrainer',
                        choices=['torchCPUTrainer', 'PL', 'torchDDP'],
                        help='PL (pytorch lightning trainer), torchDDP (custom ddp), torchCPUTrainer (custom cpu only)')
    parser.add_argument('--scoring_technique', default="NegSample",
                        help="Training technique for knowledge graph embedding model",
                        choices=["AllvsAll", "KvsAll", "1vsAll", "NegSample", "KvsSample"])
    parser.add_argument('--neg_ratio', type=int, default=1,
                        help='The number of negative triples generated per positive triple.')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 penalty e.g.(0.00001)')
    parser.add_argument('--input_dropout_rate', type=float, default=0.0)
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.0)
    parser.add_argument("--feature_map_dropout_rate", type=float, default=0.0)
    parser.add_argument("--normalization", type=str, default="None",
                        choices=["LayerNorm", "BatchNorm1d", None],
                        help="Normalization technique")
    parser.add_argument("--init_param", type=str, default=None, choices=["xavier_normal", None],
                        help="Initialization technique")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=0,
                        help="e.g. gradient_accumulation_steps=2 "
                             "implies that gradients are accumulated at every second mini-batch")
    parser.add_argument('--num_folds_for_cv', type=int, default=0,
                        help='Number of folds in k-fold cross validation.'
                             'If >2 ,no evaluation scenario is applied implies no evaluation.')
    parser.add_argument("--eval_model", type=str, default="train_val_test",
                        choices=["None", "train", "train_val", "train_val_test", "test"],
                        help='Evaluating link prediction performance on data splits. ')
    parser.add_argument("--save_model_at_every_epoch", type=int, default=1,
                        help='At every X number of epochs model will be saved. If None, we save 4 times.')
    parser.add_argument("--label_smoothing_rate", type=float, default=0.0, help='None for not using it.')
    parser.add_argument("--kernel_size", type=int, default=3,
                        help="Square kernel size for convolution based models.")
    parser.add_argument("--num_of_output_channels", type=int, default=2,
                        help="# of output channels in convolution")
    parser.add_argument("--num_core", type=int, default=0,
                        help='Number of cores to be used. 0 implies using single CPU')
    parser.add_argument("--random_seed", type=int, default=1,
                        help='Seed for all, see pl seed_everything().')
    parser.add_argument("--sample_triples_ratio", type=float, default=None, help='Sample input data.')
    parser.add_argument("--read_only_few", type=int, default=None,
                        help='READ only first N triples. If 0, read all.')
    parser.add_argument("--add_noise_rate", type=float, default=0.0,
                        help='Add x % of noisy triples into training dataset.')
    parser.add_argument('--p', type=int, default=0,
                        help='P for Clifford Algebra')
    parser.add_argument('--q', type=int, default=1,
                        help='Q for Clifford Algebra')
    parser.add_argument('--pykeen_model_kwargs', type=json.loads, default={})
    # WIP
    parser.add_argument("--byte_pair_encoding",
                        action="store_true",
                        help="Currently only avail. for KGE implemented within dice-embeddings.")
    parser.add_argument("--adaptive_swa",
                        action="store_true",
                        help="Adaptive stochastic weight averaging")
    # if description is None:
    #     return parser.parse_args()
    # return parser.parse_args(description)
    args = parser.parse_args(description) if description else parser.parse_args()
    # Update args with custom NNI parameters if provided
    if custom_args:
        for key, value in custom_args.items():
            setattr(args, key, value)

    return args


if __name__ == '__main__':
    params = nni.get_next_parameter()
    custom_args = {
        "embedding_dim": 2,
        "num_epochs": 50,
        # "model":params["models"]
        # "batch_size": params["batch_size"],
        # "weight_decay": params["weight_decay"],
        # "lr": params["learning_rate"]
        # "optimizer": params["optimizer"]
    }

    args = get_default_arguments(custom_args=custom_args)
    Execute(args).start()


#
# //"num_layers": {"_type": "choice", "_value": [2,3,4,5,6]},
# // "activation_function": {"_type": "choice","_value": ["tanh","relu","sigmoid"]},
# //  "num_epochs": {"_type": "choice","_value": [20,50,100]},
# //  "learning_rate": {"_type":"uniform", "_value":[0.0001, 0.01]},
# //  "optimizer": {"_type":"choice", "_value":["adam_momentum", "RMSprop", "adagrad", "adam" , "SGD"]},
# //  "batch_size": {"_type":"choice", "_value":[256, 512,1024]},
# //  "weight_decay": {"_type": "uniform","_value": [0.0001,0.1]},
# //  "dropout_rate": {"_type": "uniform","_value": [0.1,1]}

