import dgl
import pytest
import matplotlib.pyplot as plt
import numpy as np
import h5py
import networkx as nx
from scipy import spatial
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree


def reformat_edges_distances(edges, distances):

	idx_dst_pairs = [(idx, dest) for idx, destination in enumerate(edges) \
			for dest in destination if idx != dest] # do not include (a, a) pairs

	distances_per_edge = [distances[idx][np.where(edges[idx] == destination)][0] \
			for idx, destination in idx_dst_pairs]

	return idx_dst_pairs, distances_per_edge


@pytest.mark.parametrize("particular_edge", [0, 100, 2000, 100000])
def test_reformat(particular_edge):

	filename = '/cosma5/data/dp004/dc-cues1/features/halo_features_s99'

	maximum_distance = 1.

	with h5py.File(filename, "r+") as feats:


		positions = feats["Pos"][:] / 1000.0  # to Mpc

		#tree = spatial.cKDTree(positions, boxsize = feats['boxsize'].value/1000.)
		sktree = KDTree(positions)
		# TODO: periodic boundary conditions


		edges, distances = sktree.query_radius(positions, r = maximum_distance,
				                           return_distance = True)

		edges, distances = reformat_edges_distances(edges, distances)

		origin, dest = list(edges)[particular_edge]

		assert distances[particular_edge] == np.linalg.norm(positions[origin] - positions[dest])

		assert np.sum(np.array(distances) > maximum_distance) == 0




def hdf52graph(filename, maximum_distance, n_neighbors=None):

	with h5py.File(filename, "r+") as feats:

		positions = feats["Pos"][:] / 1000.0  # to Mpc

		#tree = spatial.cKDTree(positions, boxsize = feats['boxsize'].value/1000.)
		sktree = KDTree(positions)
		# TODO: periodic boundary conditions


		edges, distances = sktree.query_radius(positions, r = maximum_distance,
				                           return_distance = True)

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
		inv_dist_sq = 1./np.array(distances)
		G.edata["inv_dist_sq"] = torch.tensor(inv_dist_sq).unsqueeze(-1).float()
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
