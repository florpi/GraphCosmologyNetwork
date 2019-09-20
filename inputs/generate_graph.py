import dgl
import pytest
import matplotlib.pyplot as plt
import numpy as np
import h5py
import networkx as nx
from scipy.spatial import cKDTree
import torch

def periodic_distance(a, b, boxsize):
	'''
	Computes distance between vectors a and b in a periodic box
	Inputs: a and b, 3d vectors
	Outputs: dists, distance once periodic boundary conditions have been applied
	'''

	bounds = boxsize * np.ones(3)

	min_dists = np.min(np.dstack(((a - b) % bounds, (b - a) % bounds)), axis = 2)
	dists = np.sqrt(np.sum(min_dists ** 2, axis = 1))
	return dists


def reformat_edges_distances(edges, distances):
	# convert from scipy tree structure to sklearn tree structure
	empties = [np.where(dist != np.inf)[0] for dist in distances]
	distances = np.asarray([distances[i][empties[i]] for i in range(len(distances))])
	edges = np.asarray([edges[i][empties[i]] for i in range(len(edges))])

	# convert from sklearn tree structure to dgl structure
	idx_dst_pairs = [(idx, dest) for idx, destination in enumerate(edges) for dest in destination] # do not include (a, a) pairs


	distances_per_edge = [distances[idx][np.where(edges[idx] == destination)][0] for idx, destination in idx_dst_pairs]


	return idx_dst_pairs, distances_per_edge


@pytest.mark.parametrize("particular_edge", [0, 100, 2000, 100000])
def test_reformat(particular_edge):

	filename = '/cosma5/data/dp004/dc-cues1/features/halo_features_s99'

	maximum_distance = 1.

	with h5py.File(filename, "r+") as feats:

		positions = feats["Pos"][:].clip(min = 0.) / 1000.0  # to Mpc

		tree = cKDTree(positions, boxsize = feats['boxsize'].value/1000.)
		distances, edges = tree.query(positions, k=100, distance_upper_bound=maximum_distance)

		edges, distances = reformat_edges_distances(edges, distances)

		n_zero_distance = np.asarray(distances).shape[0] - np.nonzero(np.asarray(distances))[0].shape[0]

		assert n_zero_distance == len(positions)


		origin, dest = edges[particular_edge]

		assert distances[particular_edge] == periodic_distance(positions[origin], positions[dest], feats['boxsize'].value/1000.)

		assert np.sum(np.array(distances) > maximum_distance) == 0




def hdf52graph(filename, maximum_distance, n_neighbors=None):

	with h5py.File(filename, "r+") as feats:

		positions = feats["Pos"][:].clip(min = 0.) / 1000.0  # to Mpc

		tree = cKDTree(positions, boxsize = feats['boxsize'].value/1000.)
		distances, edges = tree.query(positions, k=100, distance_upper_bound=maximum_distance)
		edges, distances = reformat_edges_distances(edges, distances)

		src, dst = zip(*edges)

		G = dgl.DGLGraph()
		G.add_nodes(len(positions))
		G.add_edges(src, dst)

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

		G.ndata["feat"] = features

		inv_dist = np.where(np.isfinite(1./np.array(distances)), 1./np.array(distances), 1.)
		
		assert np.sum((inv_dist == 1.)) == len(positions)

		G.edata["inv_dist"] = torch.tensor(inv_dist).unsqueeze(-1).float()

		
		assert np.alltrue(np.isfinite(G.edata["inv_dist"]) )

		#G.edata["dist"] = torch.randn((G.number_of_edges(), 1))

		#labels = torch.tensor(feats["Ngals"][:]).float()
		#labels = labels.unsqueeze(-1)	# needed for regression
		labels = feats["Ngals"][:]
		labels = labels > 0
		labels = torch.tensor(labels).long()
		print('Graph generated!')

	return labels, G

def data2graph(filename, maximum_distance, n_neighbors=None):

	with h5py.File(filename, "r+") as feats:

		positions = feats["Pos"][:].clip(min = 0.) / 1000.0  # to Mpc

		tree = cKDTree(positions, boxsize = feats['boxsize'].value/1000.)
		distances, edges = tree.query(positions, k=100, distance_upper_bound=maximum_distance)
		edges, distances = reformat_edges_distances(edges, distances)

		src, dst = zip(*edges)

		G = dgl.DGLGraph()
		G.add_nodes(len(positions))
		G.add_edges(src, dst)

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

		G.ndata["feat"] = features

		inv_dist = np.where(np.isfinite(1./np.array(distances)), 1./np.array(distances), 1.)
		
		assert np.sum((inv_dist == 1.)) == len(positions)

		G.edata["inv_dist"] = torch.tensor(inv_dist).unsqueeze(-1).float()

		
		assert np.alltrue(np.isfinite(G.edata["inv_dist"]) )

		#G.edata["dist"] = torch.randn((G.number_of_edges(), 1))

		#labels = torch.tensor(feats["Ngals"][:]).float()
		#labels = labels.unsqueeze(-1)	# needed for regression
		labels = feats["Ngals"][:]
		labels = labels > 0
		labels = torch.tensor(labels).long()
		print('Graph generated!')

	return labels, G




if __name__ == '__main__':

	hdf5_filename = "/cosma5/data/dp004/dc-cues1/features/halo_features_s99"

	l, G = hdf52graph(hdf5_filename, 5.)

	assert np.alltrue(np.isfinite(G.edata["inv_dist"]) )



	# origin 0 print all destinations
	print(G.edges()[1][G.edges()[0] == 0])
	print(G.edges()[1][G.edges()[0] == 1])
	print(G.edges()[1][G.edges()[0] == 100])
	print(G.edges()[1][G.edges()[0] == 2000])
