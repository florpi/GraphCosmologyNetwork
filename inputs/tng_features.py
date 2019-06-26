import sys
sys.path.insert(0,'../../arepo_hdf5_library')
import read_hdf5
import numpy as np
from scipy.spatial import distance as fast_distance
from scipy.optimize import curve_fit
import h5py

class HaloCatalog:
	def __init__(self, snapnum = 99):
		'''
		Class to read halo catalogs from simulation

		snapnum: snapshot number to read

		'''
		
		# Read snapshot

		h5_dir = '/cosma7/data/TNG/TNG300-1/'
		self.snapnum = snapnum
		self.snapshot = read_hdf5.snapshot(snapnum, h5_dir)

		self.boxsize = self.snapshot.header.boxsize #kpc

		# Useful definitions
		self.dm = 1
		self.stars = 4
		self.dm_particle_mass =  self.snapshot.header.massarr[self.dm] * 1.e10

		self.stellar_mass_thresh = 1.e9 
		#self.halo_mass_thresh = self.dm_particle_mass * 100. # at least 100 particles
		self.halo_mass_thresh = 1.e11

		print('Minimum stellar mass : %.2E'%self.stellar_mass_thresh)
		print('Minimum DM halo mass : %.2E'%self.halo_mass_thresh)

		# Load fields that will be used
		useful_properties = ['GroupFirstSub', 'Group_M_Crit200', 'GroupNsubs', 'GroupPos', 'GroupVel',
				'Group_R_Crit200', 'SubhaloMassType','GroupMassType','SubhaloCM', 'GroupCM',
				'SubhaloMass', 'SubhaloMassInHalfRad','SubhaloSpin', 'SubhaloVelDisp', 'SubhaloVmax']


		self.snapshot.group_catalog(useful_properties)

		self.N_subhalos = (self.snapshot.cat['GroupNsubs']).astype(np.int64)
		# Get only resolved halos
		self.halo_mass_cut = self.snapshot.cat['Group_M_Crit200'][:] > self.halo_mass_thresh



		#self.group_offset = (np.cumsum(self.N_particles) - self.N_particles).astype(np.int64)
		#self.group_offset = self.group_offset[self.halo_mass_cut]

		self.subhalo_offset = (np.cumsum(self.N_subhalos) - self.N_subhalos).astype(np.int64)
		self.subhalo_offset = self.subhalo_offset[self.halo_mass_cut]

		self.N_subhalos = self.N_subhalos[self.halo_mass_cut]

		self.N_halos = self.N_subhalos.shape[0]
		print('%d resolved halos found.'%self.N_halos)

		self.N_gals, self.M_stars = self.Number_of_galaxies()
		print('%d resolved galaxies found.'%np.sum(self.N_gals))

		self.logM_stars = np.log10(self.M_stars)

		self.load_inmidiate_features()

		self.compute_fsub_unbound()
		self.compute_x_offset()
		

	
	def Number_of_galaxies(self):
		'''

		Given the halo catalog computes the stellar mass of a given halo, and its number of galaxies. 
		The number of galaxies is defined as the number of subhalos that halo has over a given stellar mass 
		defined inside the class
		Outputs: N_gals, number of galaxies belonging to the halo
				M_stars, mass of the stellar component boud to the halo

		'''
		# Galaxies defined as subhaloes with a stellar mass larger than the threshold
		N_gals = np.zeros((self.N_halos), dtype = np.int)
		M_stars = np.zeros((self.N_halos), dtype = np.int)
		for i in range(self.N_halos):
			N_gals[i] = np.sum(self.snapshot.cat['SubhaloMassType'][self.subhalo_offset[i]:self.subhalo_offset[i] + \
					self.N_subhalos[i], self.stars] > self.stellar_mass_thresh)
			M_stars[i] = np.sum(self.snapshot.cat['SubhaloMassType'][self.subhalo_offset[i]:self.subhalo_offset[i] + \
					self.N_subhalos[i],self.stars])

		return N_gals, M_stars
	
	def load_inmidiate_features(self):
		'''

		Loads features already computed by SUBFIND 
		+ Bullock spin parameter (http://iopscience.iop.org/article/10.1086/321477/fulltext/52951.text.html)

		'''
		self.m200c = self.snapshot.cat['Group_M_Crit200'][self.halo_mass_cut]
		self.r200c = self.snapshot.cat['Group_R_Crit200'][self.halo_mass_cut]
		self.total_mass = self.snapshot.cat['GroupMassType'][self.halo_mass_cut, self.dm]
		self.halopos = self.snapshot.cat['GroupPos'][self.halo_mass_cut]
		self.firstsub = (self.snapshot.cat['GroupFirstSub'][self.halo_mass_cut]).astype(int)
		self.bound_mass = self.snapshot.cat['SubhaloMassType'][self.firstsub, self.dm]
		self.halocm = self.snapshot.cat['SubhaloCM'][self.firstsub]
		self.fofcm = self.snapshot.cat['GroupCM'][self.halo_mass_cut]
		self.vdisp = self.snapshot.cat['SubhaloVelDisp'][self.firstsub]
		self.vmax = self.snapshot.cat['SubhaloVmax'][self.firstsub]
		#self.rmax = self.snapshot.cat['SubhaloVmaxRad'][self.firstsub]
		#self.rhalf = self.snapshot.cat['SubhaloHalfmassRadType'][self.firstsub, self.dm]
		self.spin_3d = self.snapshot.cat['SubhaloSpin'][self.firstsub,:]
		self.v200c = np.sqrt(self.snapshot.const.G * self.m200c / self.r200c/1000.) * self.snapshot.const.Mpc / 1000. 
		#km/s
		self.spin = (np.linalg.norm(self.spin_3d, axis=1)/3.) / np.sqrt(2) / self.r200c /self.v200c

		self.stellar_mass = self.self.snapshot.cat['SubhaloMassType'][self.firstsub, self.stars]

	def compute_x_offset(self):
		'''

		Computes relaxadness parameter, which is the offset between the halo center of mass and its most bound particle 
		position in units of r200c
		http://arxiv.org/abs/0706.2919

		'''

		self.x_offset = self.periodic_distance(self.fofcm, self.halopos)/self.r200c
	
	def compute_fsub_unbound(self):
		'''

		Computes another measure of how relaxed is the halo, defined as the ration between mass bound to the halo and 
		mass belonging to its FoF group

		'''

		self.fsub_unbound = 1. - self.bound_mass/self.total_mass
		

	def Concentration_from_nfw(self):
		'''

		Fit NFW profile to the halo density profile of the dark matter particles to obtain r200c/rs. 
		Procedure defined in http://arxiv.org/abs/1104.5130.
		Outputs: concentration, defined as r200c/rs
				chi2_concentration, chi2 that determines goodness of fit

		'''
		# fit an NFW profile to the halo density profile from particle data to obtain r200c/rs
		def nfw(r, rho, c):
			return np.log10(rho / ( r*c * (1. + r*c)**2))

		def density_profile(coordinates, halopos, r200c):
			r = np.linalg.norm((coordinates - halopos), axis = 1)/r200c # dimensionless
			r_bins = np.logspace(-2.5,0.,32)
			#r_bins = np.logspace(np.log10(0.05),0.,20)
			number_particles, r_edges = np.histogram(r, bins=r_bins)
			r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])
			volume = 4./3. * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)
			density = self.snapshot.header.massarr[1] * 1.e10 * number_particles / volume
			# Fit only in bins where there are particles
			r_centers = r_centers[density > 0.]
			density = density[density > 0.]
			return density/self.snapshot.const.rho_crit, r_centers

		concentration = np.zeros(self.N_halos)
		chi2_concentration = np.zeros(self.N_halos)
		for i in range(self.N_halos):
			coord = self.coordinates[self.group_offset[i] : self.group_offset[i] + self.N_particles[i]]
			density, bin_centers = density_profile(coord, self.halopos[i], self.r200c[i])
			try:
				popt, pcov = curve_fit(nfw, bin_centers, np.log10(density) )
				concentration[i] = popt[1]
				chi2_concentration[i] = 1/len(bin_centers) * np.sum( ( np.log10(density)  - nfw(bin_centers ,*popt) ) **2 )
			except:
				concentration[i] = -1.
				chi2_concentration[i] = -1.
		return concentration, chi2_concentration

	def Environment_haas(self,f):
		'''

		Measure of environment that is not correlated with host halo mass http://arxiv.org/abs/1103.0547.
		Outputs: haas_env, distance to the closest neighbor with a mass larger than f * m200c, divided by its r200c 

		'''

		haas_env = np.zeros(self.N_halos)

		def closest_node(node, nodes):
			return fast_distance.cdist([node], nodes).argmin()
	
		for i in range(self.N_halos):
			halopos_exclude = np.delete(self.halopos,i,axis=0)
			m200c_exclude = np.delete(self.m200c,i)

			halopos_neighbors = halopos_exclude[(m200c_exclude >= f *self.m200c[i])]
			if(halopos_neighbors.shape[0] == 0):
				haas_env[i] = -1.
				continue
			index_closest = closest_node(self.halopos[i], halopos_neighbors)
			distance_fneigh = np.linalg.norm(self.halopos[i] - halopos_neighbors[index_closest])

			r200c_exclude = np.delete(self.r200c,i)
			r200c_neighbor = r200c_exclude[(m200c_exclude >= f*self.m200c[i])][index_closest]
			haas_env[i] = distance_fneigh / r200c_neighbor

		return haas_env

	def total_fsub(self):
		'''

		Fraction of mass bound to substructure compared to the halo mass.
		Outputs: fsub, ratio of M_fof/M_bound

		'''
		fsub = np.zeros((self.N_halos))
		for i in range(self.N_halos):
			fsub[i] = np.sum(self.snapshot.cat['SubhaloMassType']\
					[self.subhalo_offset[i] + 1 : self.subhalo_offset[i] + self.N_particles[i], :])/self.m200c[i]

		return fsub

	def periodic_distance(self,a,b):
		'''

		Computes distance between vectors a and b in a periodic box
		Inputs: a and b, 3d vectors
		Outputs: dists, distance once periodic boundary conditions have been applied

		'''

		bounds = self.boxsize * np.ones(3)

		min_dists = np.min(np.dstack(((a - b) % bounds, (b - a) % bounds)), axis = 2)
		dists = np.sqrt(np.sum(min_dists ** 2, axis = 1))
		return dists

	def halo_shape(self):
		'''

		Describes the shape of the halo
		http://arxiv.org/abs/1611.07991

		'''
		inner = 0.15 # 0.15 *r200c (inner halo)
		outer = 1.
		self.inner_q = np.zeros(self.N_halos)
		self.inner_s = np.zeros(self.N_halos)
		self.outer_q = np.zeros(self.N_halos)
		self.outer_s = np.zeros(self.N_halos)
		for i in range(self.N_halos):
			coordinates_halo = self.coordinates[self.group_offset[i] : self.group_offset[i] + self.N_particles[i],:]
			distance = (coordinates_halo - self.halopos[i])/self.r200c[i]
			self.inner_q[i],self.inner_s[i], _, _  = ellipsoid.ellipsoidfit(distance,\
					self.r200c[i], 0,inner,weighted=True)
			self.outer_q[i],self.outer_s[i], _, _  = ellipsoid.ellipsoidfit(distance,\
					self.r200c[i], 0,outer,weighted=True)


	def save_features(self):

		output_dir = '/cosma5/data/dp004/dc-cues1/features/'
		self.output_filename = output_dir + f'halo_features_s{self.snapnum:2d}'
		print(f'Saving their properties into {self.output_filename}')

		hf = h5py.File(self.output_filename, 'w')

		hf.create_dataset('boxsize', data = self.boxsize)
		hf.create_dataset('Ngals', data = self.N_gals)
		hf.create_dataset('N_subhalos', data= self.N_subhalos)
		hf.create_dataset('M200c', data = self.m200c)
		hf.create_dataset('R200c', data = self.r200c)
		hf.create_dataset('Pos', data = self.halopos)
		hf.create_dataset('VelDisp', data = self.vdisp)
		hf.create_dataset('Vmax', data = self.vmax)
		hf.create_dataset('Spin', data = self.spin)
		hf.create_dataset('Fsub', data = self.fsub_unbound)
		hf.create_dataset('x_offset', data = self.x_offset)
		hf.close()


if __name__ == "__main__":

	halocat = HaloCatalog()
	halocat.save_features()
