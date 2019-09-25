import pandas as pd
import h5py
import numpy as np


additional_data_path = '/cosma5/data/dp004/hvrn44/HOD/'
output_file = 'merged_dataframe.h5'
data_path = '/cosma6/data/dp004/dc-cues1/tng_dataframes/'
dmo_file =  'dmo_halos.hdf5'
hydro_file = 'hydro_galaxies.hdf5'
matching_file = 'MatchedHaloes_L205n2500.dat'
additional_properties_file = 'HaloProfiles_DMO_z0.00_ext.hdf5'
mergertree_file = 'MergerTree_L205n2500TNG_DM_ext_New.hdf5'
halo_mass_cut = 1.e11

# ------------------ Halo matching between dmo and hydro simulations

matching_df = pd.read_csv(additional_data_path + matching_file,
                         delimiter = ' ', skiprows = 1,
		        names = ['ID_DMO', 'ID_HYDRO', 'M200_DMO', 'M200_HYDRO'])

# Apply mass cut

mass_matching_df = matching_df.loc[matching_df['M200_DMO'] > halo_mass_cut]

# Select only those with one-to-one mapping

idx, count = np.unique(mass_matching_df.ID_DMO, return_counts = True)

unique_dmo_idx = idx[count == 1]
unique_matching_df = mass_matching_df.loc[mass_matching_df.ID_DMO.isin(unique_dmo_idx)]

# ----------- Read in additionally halo properties from the dmo simulation

with h5py.File(additional_data_path + additional_properties_file,  'r') as hf:
        
    mass = hf['Haloes']['M200'][:]
    rmax = hf['Haloes']['Rmax'][:]
    r200c = hf['Haloes']['R200'][:]
    cnfw = hf['Haloes']['Cnfw'][:]
    rhosnfw = hf['Haloes']['Rhosnfw'][:]

    massprofile = hf['Haloes']['DMMassProfile'][:]
    #parametrized_cnfw = vmax/(rmax * 70)
    properties_ids = hf['Haloes']['GroupNumber'][:]

properties = np.vstack([properties_ids,mass, rmax,
                                     r200c, cnfw, rhosnfw]).T


properties_df = pd.DataFrame(data = properties,
                             columns = ['ID_DMO', 'M200c', 'Rmax', 
                                        'R200c', 'Cnfw', 'Rhosnfw'])

merged_matching_df = pd.merge(unique_matching_df, properties_df, on = ['ID_DMO'], how = 'inner')

 
# ----------- Read in properties from the merger trees 
with h5py.File(additional_data_path + mergertree_file, 'r') as hf:

    formation_time = hf['Haloes']['z0p50'][:]
    n_mergers = hf['Haloes']['NMerg'][:]
    mass_peak = hf['Haloes']['Mpeak'][:]
    vpeak = hf['Haloes']['Vpeak'][:]
    mergertree_ids = hf['Haloes']['Index'][:]

mergertree_data = np.vstack([mergertree_ids, formation_time, n_mergers,
                            mass_peak, vpeak]).T

mergertree_df = pd.DataFrame(data = mergertree_data, 
                columns = ['ID_DMO', 'Formation Time', 'Nmergers','MassPeak', 'vpeak'])

merged_tree_df = pd.merge(merged_matching_df, mergertree_df, on = ['ID_DMO'], how = 'inner')

# ----------- Read in properties from tng 

dmo_df = pd.read_hdf(data_path + dmo_file)

dmo_merged_df = pd.merge(merged_tree_df, dmo_df, on = ['ID_DMO'], how = 'inner')

# Check the merged haloes are the correct ones
np.testing.assert_allclose(dmo_merged_df.M200_DMO, dmo_merged_df.Group_M_Crit200, rtol = 1e-3)
dmo_merged_df = dmo_merged_df.drop(columns = ['Group_M_Crit200'])
np.testing.assert_allclose(dmo_merged_df.R200c, dmo_merged_df.Group_R_Crit200, rtol = 1e-3)
dmo_merged_df = dmo_merged_df.drop(columns = ['Group_R_Crit200'])


hydro_df = pd.read_hdf(data_path + hydro_file)

hydro_merged_df = pd.merge(dmo_merged_df, hydro_df, on = ['ID_HYDRO'], how = 'inner', suffixes = ('_dmo', '_hydro'))

np.testing.assert_allclose(hydro_merged_df.M200_HYDRO, hydro_merged_df.Group_M_Crit200, rtol = 1e-3)
hydro_merged_df = hydro_merged_df.drop(columns = ['Group_M_Crit200'])

# Save final dataframe!

print(f'Saving final dataframe into {data_path + output_file}')

hydro_merged_df.to_hdf(data_path + output_file, key = 'df', mode = 'w')
