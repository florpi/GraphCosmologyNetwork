import h5py
import pandas as pd
import numpy as np
from typing import Any, Callable
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE


def get_data(hdf5_filename: str, arg_label: str):

    # Load SubFind featured
    with h5py.File(hdf5_filename, "r+") as feats:

        features = np.column_stack(
            [
                feats["M200c"][:],
                feats["R200c"][:],
                feats["N_subhalos"][:],
                feats["VelDisp"][:],
                feats["Vmax"][:],
                feats["Spin"][:],
                feats["Fsub"][:],
                feats["x_offset"][:],
            ]
        )

        positions = feats["Pos"][:] / 1000.0
        boxsize = feats["boxsize"].value / 1000.0  # to Mpc

        # Chose label
        if arg_label == "nr_of_galaxies":
            labels = feats["Ngals"][:]
            labels = labels > 0

    # Test, train, validation split
    train_idx, test_idx, val_idx = split.train_test_val_split(
        labels.shape[0], train_size=0.5
    )
    test_idx = np.concatenate((test_idx, val_idx))  # TODO: only temporary
    train = {"features": features[train_idx, :], "labels": labels[train_idx]}
    test = {"features": features[test_idx, :], "labels": labels[test_idx]}

    return train, test


def train_test_val_split(n_nodes, train_size=0.7):
    np.random.seed(0)

    test_val_idx = np.random.choice(
        n_nodes, size=int((1.0 - train_size) * n_nodes), replace=False
    )
    val_idx = np.random.choice(
        test_val_idx, size=int(0.5 * len(test_val_idx)), replace=False
    )

    test_idx = np.array(list(set(test_val_idx) - set(val_idx)))
    train_idx = np.array(list(set(np.arange(0, n_nodes)) - set(test_val_idx)))

    assert n_nodes == len(val_idx) + len(test_idx) + len(train_idx)

    return train_idx, test_idx, val_idx


def balance_dataset(
    df_train, center_transition: float, end_transition: float, arg_sampling: str
):
    df_train_sample = _balance_df_given_mass(
        df_train, "labels", 0.0, center_transition, 0, 1, mode=arg_sampling
    )
    df_train_sample = _balance_df_given_mass(
        df_train_sample,
        "labels",
        center_transition,
        end_transition,
        1,
        0,
        mode=arg_sampling,
    )

    return df_train_sample


def _balance_df_given_mass(
    df, labels_name, minimum_mass, maximum_mass, majority, minority, mode="upsample"
):
    """
    internal function indicated by leading _
    """

    mass_threshold = (df.M200c > minimum_mass) & (df.M200c < maximum_mass)

    df_M = df[mass_threshold]

    df_M_majority = df_M[df_M[labels_name] == majority]
    df_M_minority = df_M[df_M[labels_name] == minority]

    if mode == "upsample":
        df_M_minority_upsampled = resample(
            df_M_minority,
            replace=True,
            n_samples=df_M_majority.shape[0],
            random_state=123,
        )
        return pd.concat([df_M_majority, df_M_minority_upsampled, df[~mass_threshold]])

    elif mode == "downsample":
        df_M_majority_downsampled = resample(
            df_M_majority,
            replace=False,
            n_samples=df_M_minority.shape[0],
            random_state=123,
        )
        return pd.concat(
            [df_M_majority_downsampled, df_M_minority, df[~mass_threshold]]
        )

    # elif mode == 'smote':
    #   sm = SMOTE(random_state = 12, ratio = 1.)
