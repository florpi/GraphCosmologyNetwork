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

from GNN.utils.datautils import get_data, balance_dataset, find_transition_regions
from GNN.utils.config import load_config

from sacred import Experiment
import logging

# -----------------------------------------------------------------------------
# Loggings
# -----------------------------------------------------------------------------
tag_datetime = datetime.now().strftime("%H%M_%d%m%Y")

ex = Experiment("sacred_%s" % tag_datetime, interactive=False)

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
        "model": "rnf",  # ["rnf, xgboost, lightgbm]
        "label": "dark_or_light",  # [dark_or_light, nr_of_galaxies, central_or_satellite, ..]
        "sampling": "upsample",  # [upsample, downsample]
        "use_pca": False,
    }
    ex.add_config(config_gen)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------
@ex.automain
def main(model, label, sampling, use_pca):

    logging.info("")
    logging.info(f"GROWING TREES")
    logging.info("")

    # ML-model settings
    config_file_path = "experiments/config_sklearn/config_%s.json" % model
    config = load_config(config_file_path=config_file_path)

    # -------------------------------------------------------------------------
    # Load and prepare datasets
    # -------------------------------------------------------------------------

    # Load dataset
    output_file = "merged_dataframe.h5"
    data_path = "/cosma6/data/dp004/dc-cues1/tng_dataframes/"
    hdf5_filename = data_path + output_file
    train, test = get_data(hdf5_filename, label)

    # Prepare datasets
    ## Balance training set in the transition region
    center_transition, end_transition = find_transition_regions(train)

    ex.log_scalar(
        "The labels before balancing are as follows:", train.labels.value_counts()
    )
    train = balance_dataset(train, center_transition, end_transition, sampling)
    ex.log_scalar(
        "The labels after balancing are as follows:\n a)",
        train[train.M200c < center_transition].labels.value_counts(),
    )
    ex.log_scalar(
        "b)",
        train[
            (train.M200c > center_transition) & (train.M200c < end_transition)
        ].labels.value_counts(),
    )

    # split in features and labels and convert pd.Dataframe to np.array
    train_features = train.drop(columns="labels").values
    train_labels = train["labels"].values

    test_features = test.drop(columns="labels").values
    test_labels = test["labels"].values

    ## Standarize features
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    
    if use_pca == True:
        print("Do PCA +++++++++++++++++++")
        # Create a PCA objec
        pcas = PCA(n_components=train_features.shape[1])
        test_features = pcas.fit_transform(test_features)

    # -------------------------------------------------------------------------
    # Set-up and Run random-forest (RNF) model
    # -------------------------------------------------------------------------

    rf = RandomForestClassifier(**config["model"]["parameters"])
    rf.fit(train_features, train_labels)

    # Run RNF
    test_pred = rf.predict(test_features)

    # Save results
    fname_out = "./outputs/train_%s_%s" % (model, tag_datetime)
    #train_labels.to_hdf(fname_out, key="df", mode="w")
    np.save(fname_out, train_labels)

    fname_out = "./outputs/test_%s_%s" % (model, tag_datetime)
    #test_labels.to_hdf(fname_out, key="df", mode="w")
    np.save(fname_out, test_labels)

    fname_out = "./outputs/predic_%s_%s" % (model, tag_datetime)
    np.save(fname_out, test_pred)
