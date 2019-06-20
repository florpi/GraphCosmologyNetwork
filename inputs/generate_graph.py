import dgl
import matplotlib.pyplot as plt
import numpy as np
import h5py
import networkx as nx
from scipy import spatial
import torch
from sklearn.preprocessing import StandardScaler


def hdf52graph(filename, maximum_distance):

	with h5py.File(filename,'r+') as feats: 

		positions = feats['Pos'][:]/1000. # to Mpc

		#TODO; FIX PERIODIC BOX
		#tree = spatial.cKDTree(positions, boxsize = feats['boxsize'].value/1000.)
		tree = spatial.cKDTree(positions)

		edgeList = tree.query_pairs(maximum_distance) 

		src, dst = zip(*edgeList)
		G = dgl.DGLGraph()
		G.add_nodes(len(positions))
		G.add_edges(src, dst)

		features = np.column_stack([feats['M200c'][:], 
							feats['R200c'][:],
							feats['N_subhalos'][:],
							feats['VelDisp'][:],
							feats['Vmax'][:],
							feats['Spin'][:],
							feats['Fsub'][:],
							feats['x_offset'][:]])


		G.ndata['feat'] = features

		G.ndata['std_feat'] = features

		labels = torch.tensor(feats['Ngals'][:]).float()
		labels = labels.unsqueeze(-1) # needed for regression


	return labels, G



