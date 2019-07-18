import dgl
import matplotlib.pyplot as plt
import numpy as np
import h5py
import networkx as nx
from scipy import spatial
import torch
from sklearn.preprocessing import StandardScaler


def hdf52graph(filename, maximum_distance, n_neighbors=None):

    with h5py.File(filename, "r+") as feats:

        positions = feats["Pos"][:] / 1000.0  # to Mpc

        tree = spatial.cKDTree(positions, boxsize = feats['boxsize'].value/1000.)

        # TODO; check edge-criterion, e.g.: distance, grav.-force
        edgeList = tree.query_pairs(maximum_distance)
        # distances, edgeList = tree.query(positions,n_neighbors, distance_upper_bound = maximum_distance)

        src, dst = zip(*edgeList)
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

        #labels = torch.tensor(feats["Ngals"][:]).float()
        #labels = labels.unsqueeze(-1)  # needed for regression
        labels = feats["Ngals"][:]
        labels = labels > 0
        labels = torch.tensor(labels).long()

    return labels, G


if __name__ == '__main__':

    hdf5_filename = "/cosma5/data/dp004/dc-cues1/features/halo_features_s99"

    l, G = hdf52graph(hdf5_filename, 5.)
