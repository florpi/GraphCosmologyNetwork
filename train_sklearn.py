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
# Loggings 
# -----------------------------------------------------------------------------
tag_datetime = datetime.now().strftime("%H%M_%d%m%Y")

ex = Experiment(
    "sacred_%s" % tag_datetime,
    #interactive=False,
)

logging.basicConfig(
    filename="experiments/logs/log_%s.log" % tag_datetime,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

# -----------------------------------------------------------------------------
# General Settings/Command-line settings 
# -----------------------------------------------------------------------------
@ex.config
def cfg():
    # General settings
    config_gen = {
        "model": 'rnf', #["rnf, xgboost, lightgbm] 
        "label": 'dark_or_light', #[dark_or_light, nr_of_galaxies, central_or_satellite, ..]
        "sampling": 'upsampling',  #[upsampling, downsampling] 
        "PCA": False, 
    } 
    ex.add_config(config_gen)
    
# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------
@ex.automain
def main(model, label, sampling, PCA):
    logging.info("")
    logging.info(f"GROWING TREES")
    logging.info("")

    # Read in command line arguments
    args = get_arguments()
    
    # ML-model settings
    config_file_path = "experiments/config_sklearn/config_%s.json" % model
    config = load_config(config_file_path=config_file_path)

    # -------------------------------------------------------------------------
    # Load and prepare datasets
    # -------------------------------------------------------------------------

    # Load dataset
    output_file = 'merged_dataframe.h5'                                          
    data_path = '/cosma6/data/dp004/dc-cues1/tng_dataframes/'
    hdf5_filename = data_path + output_file 
    train, test = get_data(hdf5_filename, label)

    # Prepare datasets
    ## Balance training set in the transition region
    #TODO: find these params automatically
    center_transition = 2.1e11
    end_transition = 8e11

    train_features = train.drop(columns="labels")
    train_labels = train["labels"]
    
    test_features = test.drop(columns="labels")
    test_labels = test["labels"]

    ## Standarize features
    scaler = StandardScaler()
    scaler.fit(train_features)
    std_train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    if PCA is True:
        # n_comp = 7
        # pca = PCA(n_components=n_comp)
        # pca = PCA().fit(std_train_features)
        pca_data = PCA().fit_transform(std_train_features)
        pca_inv_data = PCA().inverse_transform(np.eye(len(feature_names)))

    # -------------------------------------------------------------------------
    # Set-up and Run random-forest (RNF) model
    # -------------------------------------------------------------------------

    rf = RandomForestClassifier(**config["model"]["parameters"])
    rf.fit(train_features, train_labels)

    # Run RNF
    test_pred = rf.predict(test_features)

    # ex.log_scalar("true_cancel_count", true_cancel_count)  # <- sacred decorator
    # ex.log_scalar("pred_cancel_count", pred_cancel_count)  # <- sacred decorator
    # ex.log_scalar("train_cancel_orders", train_cancel_count)  # <- sacred decorator

    # Save result
    fname_out = "./outputs/%s_%s" % (model, tag_datetime)
    np.save(fname_out, test_pred)
