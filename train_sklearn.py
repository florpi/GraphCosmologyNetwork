import os, time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Callable

import numpy as np

import pickle

from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    precision_recall_fscore_support,
)

from GNN.utils.datautils import get_data, balance_dataset
from GNN.utils.config import load_config

# from GNN.utils.importing import get_class_by_name

from sacred import Experiment
import logging

# -----------------------------------------------------------------------------
# Logging and Experiment set-up
# -----------------------------------------------------------------------------

tag_datetime = datetime.now().strftime("%H%M_%d%m%Y")

ex = Experiment("logs/sacred_%s.log" % tag_datetime)

logging.basicConfig(
    filename="experiments/logs/log_%s.log" % tag_datetime,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------


#@ex.config  # <- sacred decorator
def get_arguments() -> argparse.Namespace:
    """
	Set up an ArgumentParser to get the command line arguments.

	Returns:
		A Namespace object containing all the command line arguments
		for the script.
	"""

    # Set up parser
    parser = argparse.ArgumentParser()

    # Add arguments
    # TODO:
    parser.add_argument(
        "--model",
        action="store_true",
        default="rnf",
        help=(
            "Which non-convolutional ML-model do you want to use: "
            + "rnf, xgboost, lightgbm? Default: rnf."
        ),
    )
    # TODO:
    parser.add_argument(
        "--label",
        action="store_true",
        default="nr_of_galaxies",
        help="What to learn: dark_or_light, nr_of_galaxies, central_or_satellite, ...",
    )
    parser.add_argument(
        "--sampling",
        default="upsampling",
        type=str,
        help="Balance training set via: upsampling, downsampling. Default: upsampling.",
    )
    # TODO:
    parser.add_argument(
        "--sacred",
        action="store_true",
        default=False,
        help="Use Sacred to manage experiments? Default: False.",
    )
    # TODO:
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=False,
        help="Use TensorBoard to log training progress? Default: False.",
    )
    parser.add_argument(
        "--n_estimators",
        default=500,
        type=int,
        metavar="N",
        help="The number of trees in the forest. Default: 500.",
    )
    # TODO:
    parser.add_argument(
        "--PCA",
        action="store_true",
        default=False,
        help="Which features shoud be used: input file or PCA analysis? Default: input file.",
    )
    parser.add_argument(
        "--experiment",
        default="config_sklearn",
        type=str,
        metavar="PATH",
        help="Name of the experiment to run (must be a folder "
        'in the experiments dir). Default: "config_sklearn".',
    )

    # Parse and return the arguments (as a Namespace object)
    arguments = parser.parse_args()
    return arguments


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------


#@ex.automain  # <- sacred decorator
#def run():
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------
    # Read in command line arguments
    args = get_arguments()

    logging.info("")
    logging.info(f"GROWING TREES")
    logging.info("")

    # -------------------------------------------------------------------------
    # Load the experiment configuration
    # -------------------------------------------------------------------------

    # Construct the path to the experiment config file
    experiment_dir = os.path.join("experiments", args.experiment)
    config_file_path = os.path.join(experiment_dir, "config_%s.json" % args.model)

    # Load the config
    config = load_config(config_file_path=config_file_path)

    # -------------------------------------------------------------------------
    # Load and prepare datasets
    # -------------------------------------------------------------------------

    # Load dataset
    output_file = 'merged_dataframe.h5'                                          
    data_path = '/cosma6/data/dp004/dc-cues1/tng_dataframes/'
    hdf5_filename = data_path + output_file 
    train, test = get_data(hdf5_filename, args.label)

    # Prepare datasets
    ## Balance training set in the transition region
    center_transition = 2.1e11
    end_transition = 8e11

    logging.info(
        "The labels before balancing are as follows:", train.labels.value_counts()
    )
    train = balance_dataset(
        train, center_transition, end_transition, args.sampling
    )
    logging.info(
        "The labels after balancing are as follows:",
        train[train.M200c < center_transition].labels.value_counts(),
        train[
            (train.M200c > center_transition) & (train.M200c < end_transition)
        ].labels.value_counts(),
    )

    train_features = train.drop(columns="labels")
    train_labels = train["labels"]

    ## Standarize features
    scaler = StandardScaler()
    scaler.fit(train_features)
    std_train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    if args.PCA is True:
        # n_comp = 7
        # pca = PCA(n_components=n_comp)
        # pca = PCA().fit(std_train_features)
        pca_data = PCA().fit_transform(std_train_features)
        pca_inv_data = PCA().inverse_transform(np.eye(len(feature_names)))

    # -------------------------------------------------------------------------
    # Set-up and Run random-forest (RNF) model
    # -------------------------------------------------------------------------

    # Create instance of the RNF we want to use as specified in
    # the experiment config file
    """
    model_class = get_class_by_name(
        module_name=config["model"]["module"], class_name=config["model"]["class"]
    )
    model = model_class(**config["model"]["parameters"])
    logging.info("model: \t\t\t", model.__class__.__name__)
    """
    rf = RandomForestClassifier(n_estimators=args.n_estimators)
    rf.fit(std_train_features, df_train_labels)

    # Run RNF
    test_pred = rf.predict(test_features)

    # ex.log_scalar("true_cancel_count", true_cancel_count)  # <- sacred decorator
    # ex.log_scalar("pred_cancel_count", pred_cancel_count)  # <- sacred decorator
    # ex.log_scalar("train_cancel_orders", train_cancel_count)  # <- sacred decorator

    # Save result
    fname_out = ""
    np.save(fname_out, test_pred)
