import sys

sys.path.insert(0, "../../arepo_hdf5_library")
import read_hdf5
import numpy as np
from scipy.spatial import distance as fast_distance
from scipy.optimize import curve_fit
import h5py
import pandas as pd

class Catalog:
	'''

	Class to describe a catalog and its methods


	'''
	def __init__(self, 
			h5_dir: str, 
			snapnum: int):
		'''
		Args:
			h5_dir: directory containing tng data
			snapnum: Snapshot number to read
		'''

		self.snapnum = snapnum

		self.snapshot = read_hdf5.snapshot(snapnum, h5_dir)

		# Useful definitions
		self.dm = 1
		self.stars = 4
		self.dm_particle_mass = self.snapshot.header.massarr[self.dm] * 1.0e10
		self.output_dir = '/cosma6/data/dp004/dc-cues1/tng_dataframes/'

	def load_inmidiate_features(self, 
			group_feature_list: list, 
			sub_feature_list: list):
		"""
		Loads features already computed by SUBFIND 
		+ Bullock spin parameter (http://iopscience.iop.org/article/10.1086/321477/fulltext/52951.text.html)

		Args:
			group_feature_list: list of group features to load and save as attributes of the class
			sub_feature_list: list of subgroup features to load and save as attributes of the class

		"""

		for feature in group_feature_list:
			value = self.snapshot.cat[feature][self.halo_mass_cut]
			if ('Crit200' in feature) or ('Mass' in feature):
				value *= self.snapshot.header.hubble
			setattr(self, feature, value)

		self.firstsub = (self.GroupFirstSub).astype(int)

		self.v200c = np.sqrt(self.snapshot.const.G * self.Group_M_Crit200/ self.Group_R_Crit200/1000.) * self.snapshot.const.Mpc / 1000. 

		for feature in sub_feature_list:
			value = self.snapshot.cat[feature][self.firstsub]
			if ('Crit200' in feature) or ('Mass' in feature):
				value *= self.snapshot.header.hubble
			setattr(self, feature.replace('Subhalo', ''), value)

		self.Spin= (np.linalg.norm(self.Spin, axis=1)/3.) / np.sqrt(2) / self.Group_R_Crit200/self.v200c

		self.bound_mass = self.MassType[:, self.dm] 
		self.total_mass = self.GroupMassType[:, self.dm]


	def compute_x_offset(self):
		"""
		Computes relaxadness parameter, which is the offset between the halo center of mass and its most bound particle 
		position in units of r200c
		http://arxiv.org/abs/0706.2919

		"""

		self.x_offset = self.periodic_distance(self.GroupCM, self.GroupPos) / self.Group_R_Crit200

	def compute_fsub_unbound(self):
		"""

		Computes another measure of how relaxed is the halo, defined as the ration between mass bound to the halo and 
		mass belonging to its FoF group

		"""

		self.fsub_unbound = 1.0 - self.bound_mass / self.total_mass

	def Concentration_from_nfw(self):
		"""

		Fit NFW profile to the halo density profile of the dark matter particles to obtain r200c/rs. 
		Procedure defined in http://arxiv.org/abs/1104.5130.
		Outputs: concentration, defined as r200c/rs
				chi2_concentration, chi2 that determines goodness of fit

		"""
		# fit an NFW profile to the halo density profile from particle data to obtain r200c/rs
		def nfw(r, rho, c):
			return np.log10(rho / (r * c * (1.0 + r * c) ** 2))

		def density_profile(coordinates, halopos, r200c):
			r = np.linalg.norm((coordinates - halopos), axis=1) / r200c  # dimensionless
			r_bins = np.logspace(-2.5, 0.0, 32)
			# r_bins = np.logspace(np.log10(0.05),0.,20)
			number_particles, r_edges = np.histogram(r, bins=r_bins)
			r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])
			volume = 4.0 / 3.0 * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
			density = (
				self.snapshot.header.massarr[1] * 1.0e10 * number_particles / volume
			)
			# Fit only in bins where there are particles
			r_centers = r_centers[density > 0.0]
			density = density[density > 0.0]
			return density / self.snapshot.const.rho_crit, r_centers

		concentration = np.zeros(self.N_halos)
		chi2_concentration = np.zeros(self.N_halos)
		for i in range(self.N_halos):
			coord = self.coordinates[
				self.group_offset[i] : self.group_offset[i] + self.N_particles[i]
			]
			density, bin_centers = density_profile(
				coord, self.GroupPos[i], self.r200c[i]
			)
			try:
				popt, pcov = curve_fit(nfw, bin_centers, np.log10(density))
				concentration[i] = popt[1]
				chi2_concentration[i] = (
					1
					/ len(bin_centers)
					* np.sum((np.log10(density) - nfw(bin_centers, *popt)) ** 2)
				)
			except:
				concentration[i] = -1.0
				chi2_concentration[i] = -1.0
		return concentration, chi2_concentration

	def Environment_haas(self, f: float):
		"""

		Measure of environment that is not correlated with host halo mass http://arxiv.org/abs/1103.0547.
		Outputs: haas_env, distance to the closest neighbor with a mass larger than f * m200c, divided by its r200c 

		Args: 
			f: threshold to select minimum mass of neighbor to consider.

		"""

		haas_env = np.zeros(self.N_halos)

		def closest_node(node, nodes):
			return fast_distance.cdist([node], nodes).argmin()

		for i in range(self.N_halos):
			halopos_exclude = np.delete(self.GroupPos, i, axis=0)
			m200c_exclude = np.delete(self.m200c, i)

			halopos_neighbors = halopos_exclude[(m200c_exclude >= f * self.m200c[i])]
			if halopos_neighbors.shape[0] == 0:
				haas_env[i] = -1.0
				continue
			index_closest = closest_node(self.GroupPos[i], halopos_neighbors)
			distance_fneigh = np.linalg.norm(
				self.GroupPos[i] - halopos_neighbors[index_closest]
			)

			r200c_exclude = np.delete(self.r200c, i)
			r200c_neighbor = r200c_exclude[(m200c_exclude >= f * self.m200c[i])][
				index_closest
			]
			haas_env[i] = distance_fneigh / r200c_neighbor

		return haas_env

	def total_fsub(self):
		"""

		Fraction of mass bound to substructure compared to the halo mass.
		Outputs: fsub, ratio of M_fof/M_bound

		"""
		fsub = np.zeros((self.N_halos))
		for i in range(self.N_halos):
			fsub[i] = (
				np.sum(
					self.snapshot.cat["MassType"][
						self.subhalo_offset[i]
						+ 1 : self.subhalo_offset[i]
						+ self.N_particles[i],
						:,
					]
				)
				/ self.m200c[i]
			)

		return fsub

	def periodic_distance(self, a: np.ndarray, b: np.ndarray) -> np.array:
		"""

		Computes distance between vectors a and b in a periodic box
		Args:
			a: first array.
			b: second array.
		Returns:
			dists, distance once periodic boundary conditions have been applied

		"""

		bounds = self.boxsize * np.ones(3)

		min_dists = np.min(np.dstack(((a - b) % bounds, (b - a) % bounds)), axis=2)
		dists = np.sqrt(np.sum(min_dists ** 2, axis=1))
		return dists

	def halo_shape(self):
		"""

		Describes the shape of the halo
		http://arxiv.org/abs/1611.07991

		""" 
		inner = 0.15 # 0.15 *r200c (inner halo)
		outer = 1.
		self.inner_q = np.zeros(self.N_halos)
		self.inner_s = np.zeros(self.N_halos)
		self.outer_q = np.zeros(self.N_halos)
		self.outer_s = np.zeros(self.N_halos)
		for i in range(self.N_halos):
			coordinates_halo = self.coordinates[self.group_offset[i] : self.group_offset[i] + self.N_particles[i],:]
			distance = (coordinates_halo - self.GroupPos[i])/self.r200c[i]
			self.inner_q[i],self.inner_s[i], _, _  = ellipsoid.ellipsoidfit(distance,\
					self.r200c[i], 0,inner,weighted=True)
			self.outer_q[i],self.outer_s[i], _, _  = ellipsoid.ellipsoidfit(distance,\
					self.r200c[i], 0,outer,weighted=True)


	def save_features(self, 
			output_filename: str, 
			features_to_save: list):
		'''
		Save given features to hdf5 

		Args:
			output_filename: file to save hdf5 file.
			features_to_save: list of feature names to save into file.

		'''

		print(f'Saving their properties into {self.output_dir + output_filename}')

		if 'GroupPos' in features_to_save:
			remove_grouppos = True
		else:
			remove_grouppos = False

		feature_list = []
		for feature in features_to_save:
			if feature != 'GroupPos':
				feature_list.append(getattr(self, feature))

		feature_list = np.asarray(feature_list).T
		features_to_save.remove('GroupPos') if 'GroupPos' in features_to_save else None

		df = pd.DataFrame( data = feature_list,
				columns = features_to_save)

		if remove_grouppos:
			df['x'] = self.GroupPos[:,0]/1000. # To Mpc/h
			df['y'] = self.GroupPos[:,1]/1000.
			df['z'] = self.GroupPos[:,2]/1000.

		df.to_hdf(self.output_dir + output_filename, key = 'df', mode = 'w')




class HaloCatalog(Catalog):

	def __init__(self):
		"""
		Class to read halo catalogs from simulation

		"""

		# Read snapshot
		h5_dir = "/cosma7/data/TNG/TNG300-1-Dark/"
		super().__init__(h5_dir, 99)
		self.boxsize = self.snapshot.header.boxsize / self.snapshot.header.hubble # kpc
		self.halo_mass_thresh = 1.0e11 

		print("Minimum DM halo mass : %.2E" % self.halo_mass_thresh)

		# Load fields that will be used
		group_properties = [
			"GroupFirstSub",
			"Group_M_Crit200",
			"GroupNsubs",
			"GroupPos",
			"GroupVel",
			"Group_R_Crit200",
			"GroupMassType",
		]

		sub_properties = [
			"SubhaloMassType",
			"SubhaloCM",
			"GroupCM",
			"SubhaloMass",
			"SubhaloMassInHalfRad",
			"SubhaloSpin",
			"SubhaloVelDisp",
			"SubhaloVmax",
		]

		self.snapshot.group_catalog(group_properties + sub_properties)

		self.N_subhalos = (self.snapshot.cat["GroupNsubs"]).astype(np.int64)
		# Get only resolved halos
		self.halo_mass_cut = (
				self.snapshot.cat["Group_M_Crit200"][:] * self.snapshot.header.hubble > self.halo_mass_thresh
		)
		# Save IDs of haloes
		self.ID_DMO = np.arange(0, len(self.halo_mass_cut))
		self.ID_DMO = self.ID_DMO[self.halo_mass_cut]

		self.subhalo_offset = (np.cumsum(self.N_subhalos) - self.N_subhalos).astype(
			np.int64
		)
		self.subhalo_offset = self.subhalo_offset[self.halo_mass_cut]

		self.N_subhalos = self.N_subhalos[self.halo_mass_cut]

		self.N_halos = self.N_subhalos.shape[0]
		print("%d resolved halos found." % self.N_halos)

		self.load_inmidiate_features(group_properties, sub_properties)

		self.compute_fsub_unbound()
		self.compute_x_offset()

class GalaxyCatalog(Catalog):
	def __init__(self, snapnum=99):
		"""
		Class to read galaxy catalogs from simulation

		"""

		# Read snapshot

		h5_dir = "/cosma7/data/TNG/TNG300-1/"
		super().__init__(h5_dir, 99)
		self.boxsize = self.snapshot.header.boxsize / self.snapshot.header.hubble # kpc

		self.stellar_mass_thresh = 1.0e9 
		# Load fields that will be used
		group_properties = [
			"Group_M_Crit200",
			"GroupMassType",
			"GroupNsubs",
			"GroupPos"
		]

		sub_properties = [
			"SubhaloMassType",
		]

		self.snapshot.group_catalog(group_properties + sub_properties)


		self.halo_mass_cut = (
				self.snapshot.cat["Group_M_Crit200"][:] * self.snapshot.header.hubble > 0. 
		)

		N_halos_all = self.snapshot.cat['Group_M_Crit200'].shape[0]

		self.ID_HYDRO = np.arange(0, N_halos_all)
		self.ID_HYDRO = self.ID_HYDRO[self.halo_mass_cut]

		self.N_halos= (self.snapshot.cat['Group_M_Crit200'])[self.halo_mass_cut].shape[0]

		self.N_subhalos = (self.snapshot.cat["GroupNsubs"]).astype(np.int64)
		self.subhalo_offset = (np.cumsum(self.N_subhalos) - self.N_subhalos).astype(
			np.int64
		)
		self.subhalo_offset = self.subhalo_offset[self.halo_mass_cut]
		self.N_subhalos = self.N_subhalos[self.halo_mass_cut]


		self.Group_M_Crit200 = self.snapshot.cat['Group_M_Crit200'][self.halo_mass_cut] * self.snapshot.header.hubble
		self.GroupPos = self.snapshot.cat['GroupPos'][self.halo_mass_cut, :] 
		self.N_gals, self.M_stars = self.Number_of_galaxies()
		self.logM_stars = np.log10(self.M_stars)
		print("%d resolved galaxies found." % np.sum(self.N_gals))

		print("Minimum stellar mass : %.2E" % self.stellar_mass_thresh)

	def Number_of_galaxies(self):
		"""

		Given the halo catalog computes the stellar mass of a given halo, and its number of galaxies. 
		The number of galaxies is defined as the number of subhalos that halo has over a given stellar mass 
		defined inside the class

		Returns:
				N_gals: number of galaxies belonging to the halo
				M_stars: mass of the stellar component bound to the halo

		"""
		# Subhaloes defined as galaxies with a stellar mass larger than the threshold
		N_gals = np.zeros((self.N_halos), dtype=np.int)
		M_stars = np.zeros((self.N_halos), dtype=np.int)
		for i in range(self.N_halos):
			N_gals[i] = np.sum(
				self.snapshot.cat["SubhaloMassType"][
					self.subhalo_offset[i] : self.subhalo_offset[i]
					+ self.N_subhalos[i],
					self.stars,
				]
				> self.stellar_mass_thresh
			)
			M_stars[i] = np.sum(
				self.snapshot.cat["SubhaloMassType"][
					self.subhalo_offset[i] : self.subhalo_offset[i]
					+ self.N_subhalos[i],
					self.stars,
				]
			)

		return N_gals, M_stars



if __name__ == "__main__":

	halocat = HaloCatalog()
	features_to_save = ['ID_DMO','N_subhalos', 'Group_M_Crit200', 'Group_R_Crit200',
			'VelDisp', 'Vmax', 'Spin', 'fsub_unbound', 'x_offset' , 'GroupPos']
	halocat.save_features('dmo_halos.hdf5', features_to_save)

	galcat = GalaxyCatalog()
	features_to_save = ['ID_HYDRO','N_gals', 'M_stars', 'Group_M_Crit200', 'GroupPos']
	galcat.save_features('hydro_galaxies.hdf5', features_to_save)
