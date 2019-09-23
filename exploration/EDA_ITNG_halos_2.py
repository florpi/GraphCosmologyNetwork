import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import h5py
import pandas as pd
import pandas_profiling

data_path = '/cosma5/data/dp004/hvrn44/HOD/'
halo_mass_cut = 1.e11 #solar masses

hf = h5py.File(data_path + 'HaloProfiles_DMO_z0.00_ext.hdf5', 'r')    

# Create Dataframe of integer halos values
dic = {}
for key in hf['Haloes'].keys():
    if (len(hf['Haloes'][key].shape) == 1 and
        hf['Haloes'][key].shape[0] == 229265):
        print('%s Shape : ' % key, hf['Haloes'][key].shape)
        dic[key] = hf['Haloes'][key][:]

df = pd.DataFrame(dic)

profile = df.profile_report(
    title='Pandas Profiling Report Nr. 1',
    style={'full_width':True},
)
profile.to_file(
    output_file="ITNG_DM300_hmc1e11.html",
    silent=True,
)

