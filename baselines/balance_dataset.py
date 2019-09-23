import numpy as np
import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE


def balance_df_given_mass(df, labels_name, minimum_mass, maximum_mass, majority, minority, mode = 'upsample'):

	mass_threshold = (df.M200c > minimum_mass) & (df.M200c < maximum_mass)

	df_M = df[mass_threshold]

	df_M_majority = df_M[df_M[labels_name] == majority ]
	df_M_minority = df_M[df_M[labels_name] == minority ]

	if mode == 'upsample':

		df_M_minority_upsampled = resample(df_M_minority,
				replace = True,
				n_samples = df_M_majority.shape[0],
				random_state = 123)

		return pd.concat([ df_M_majority, df_M_minority_upsampled , df[~mass_threshold] ])

	elif mode == 'downsample':

		df_M_majority_downsampled = resample(df_M_majority,
				replace = False,
				n_samples = df_M_minority.shape[0],
				random_state = 123)

		return pd.concat([ df_M_majority_downsampled, df_M_minority, df[~mass_threshold]] )


#	elif mode == 'smote':

		#sm = SMOTE(random_state = 12, ratio = 1.)

		



