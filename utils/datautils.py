import numpy as np
from typing import Any, Callable

from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

from scipy.stats import binned_statistic

import h5py
import pandas as pd

# from imblearn.over_sampling import SMOTE
from scipy.interpolate import interp1d


def get_data(hdf5_filename: str, arg_label: str):
    """
	"""
    df = pd.read_hdf(hdf5_filename, key="df", mode="r")

    # Chose label
    if arg_label == "dark_or_light":
        df["labels"] = df.N_gals > 0
        df = df.drop(
            columns=[
                "N_gals",
                "ID_HYDRO",
                "ID_DMO",
                "M200_HYDRO",
                "M_stars",
                "x_hydro",
                "y_hydro",
                "z_hydro",
            ]
        )
    elif arg_label == "nr_of_galaxies":
        df["labels"] = df.N_gals
        df = df.drop(
            columns=[
                "N_gals",
                "ID_HYDRO",
                "ID_DMO",
                "M200_HYDRO",
                "M_stars",
                "x_hydro",
                "y_hydro",
                "z_hydro",
            ]
        )
    elif arg_label == "both":
        df["labels"] = df.N_gals > 0

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


def find_transition_regions(df: pd.DataFrame):
    """

	Function to find two masses: where half the haloes are luminous,
	and where all haloes are luminous

	Args:
		df: dataframe containing masses and wheather luminous or dark
	Returns:
		mass_center: mass at which half of the haloes are luminous.
		mass_end: mass at which 100% of haloes are luminous.

	"""
    nbins = 15
    bins = np.logspace(np.log10(np.min(df.M200c)), 12.5, nbins + 1)

    nluminous, mass_edges, _ = binned_statistic(
        df.M200c, df.labels, statistic="mean", bins=bins
    )

    interpolator = interp1d(nluminous, (mass_edges[1:] + mass_edges[:-1]) / 2.0)

    mass_center = interpolator(0.5)

    mass_end = ((mass_edges[1:] + mass_edges[:-1]) / 2.0)[nluminous == 1.0][0]

    return mass_center, mass_end


def balance_dataset(
    df, center_transition: float, end_transition: float, arg_sampling: str
):
    df_sample = _balance_df_given_mass(
        df, "labels", 0.0, center_transition, 0, 1, mode=arg_sampling
    )
    df_sample = _balance_df_given_mass(
        df_sample, "labels", center_transition, end_transition, 1, 0, mode=arg_sampling
    )

    return df_sample


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
    # 	sm = SMOTE(random_state = 12, ratio = 1.)


def pca_transform(train: np.ndarray, test: np.ndarray, arg_pca):
    """
    """
    
    if arg_pca == int:
        return _pca_manual(train, test, arg_pca)
    elif arg_pca == "cross_val"
        return _pca_cross_val(train, test)

    elif arg_pca == 'mle':
        return _pca_mle(train, test)


def _pca_manual(train, test, n_components):
    """
    """
    pca = PCA(n_components=n_components)

    # Perform feature optimization
    train = pca.fit_transform(train)
    test = pca.fit_transform(test)
    
    return train, test, n_components_pca


def _pca_cross_val(train, test):
    """
    """
    pca = PCA(svd_solver='full')

    pca_scores = []
    n_components = np.arange(0, train.shape[1], 1)

    # determine cross-val. score for different dimensions of feature space
    for n in n_components:
        pca.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, train, cv=5)))

    # choose # of dimensions with highest cross-val. score
    # another option would be: pca = PCA(svd_solver='full', n_components='mle')
    n_components_pca = n_components[np.argmax(pca_scores)]
    pca.n_components = n_components_pca
    
    # Perform feature optimization
    train = pca.fit_transform(train)
    test = pca.fit_transform(test)

    return train, test, n_components_pca


def _pca_mle(train, test):
    """
    """
    pca = PCA(svd_solver='full', n_components='mle')

    # Perform feature optimization
    train = pca.fit_transform(train)
    test = pca.fit_transform(test)
    
    n_components_pca = "mle"

    return train, test, n_components_pca

