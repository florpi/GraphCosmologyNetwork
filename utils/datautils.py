import numpy as np
from typing import Any, Callable
from sklearn.utils import resample
from scipy.stats import binned_statistic

import h5py
import pandas as pd

from imblearn.over_sampling import SMOTE


def get_data(hdf5_filename: str, arg_label: str):
    """
    """
    df = pd.read_hdf(hdf5_filename, key='df', mode='r')

    # Chose label
    if arg_label == "dark_or_light":
        df['labels'] = df.N_gals > 0
        df = df.drop(columns = 'N_gals')
    elif arg_label == "nr_of_galaxies":
        df['labels'] = df.N_gals
        df = df.drop(columns = 'N_gals')

    # Test, train, validation split
    train_idx, test_idx, val_idx = _train_test_val_split(
        df.labels.values.shape[0], train_size=0.5
    )
    test_idx = np.concatenate((test_idx, val_idx))  # TODO: only temporary

    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    return train, test


def _train_test_val_split(n_nodes, train_size=0.7):
    """
    internal function indicated by leading _
    """
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
    df, center_transition: float, end_transition: float, arg_sampling: str
):
    """
    """
    
    center_transition, end_transition = _find_center_of_balance(
        df, transition_center, transition_end
    )

    # TODO: fix balance parameters; return none 
    df_sample = _balance_df_given_mass(
        df, "labels", 0.0, center_transition, 0, 1, mode=arg_sampling
    )
    df_sample = _balance_df_given_mass(
        df_sample,
        "labels",
        center_transition,
        end_transition,
        1,
        0,
        mode=arg_sampling,
    )

    return df_train_sample


def _find_center_of_balance(df):
    """
    """

    # bin data
    nbins= 15
    bins = np.logspace(
        np.log10(np.min(df.M200c)),
        12.5,
        nbins+1,
    )
    
    # nr. of luminous galaxies
    nluminous, edges, _ = binned_statistic(
        df.M200c,
        df.labels, 
        statistic='mean',
        bins=bins,
    )

    # Find x for which y = 0.5
    interpolator = interp1d(nluminous, (edges[1:]+edges[:-1])/2.)
    centre = interpolator(0.5)
    end = ((edges[1:]+edges[:-1])/2.)[nluminous == 1.][0]
    
    return centre, end


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
