import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN.inputs import generate_graph, split
from GNN.models import gcn, gat
from sklearn.preprocessing import StandardScaler
import time


# **************************** DEFINE HYPERPARAMS ***********************
maximum_distance = 2 # Mpc
learning_rate = 0.01
num_epochs = 30

# **************************** DEFINE GRAPH ***********************

hdf5_filename = '/cosma5/data/dp004/dc-cues1/features/halo_features_s99'
labels, G = generate_graph.hdf52graph(hdf5_filename, maximum_distance)


# ****************** MASKS FOR TRAIN/VAL/SPLIT ***********************

train_idx, test_idx, val_idx = split.train_test_val_split(len(G.nodes()))

train_mask = torch.Tensor(train_idx).long()
test_mask = torch.Tensor(test_idx).long()
val_mask = torch.Tensor(val_idx).long()

scaler = StandardScaler()

scaler.fit(G.ndata['feat'][train_idx,:])

G.ndata['std_feat'] = scaler.transform(G.ndata['feat'])
# Standarize features based on training set statistics


# ******************  DEFINE NETWORK ***********************

#net = gcn.GCN(G.ndata['std_feat'].shape[-1], 32, 1)
net = gat.GAT(G,
		in_dim = G.ndata['std_feat'].shape[-1],
		hidden_dim = 8,
		out_dim = 1,
		num_heads = 2)



# ******************  TRAINING LOOP ***********************

optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
all_logits, dur = [], []
for epoch in range(num_epochs):

	if epoch >= 3:
		t0 = time.time()

	#logits = net(G,  torch.tensor(G.ndata['std_feat']).float())
	logits = net(torch.tensor(G.ndata['std_feat']).float())

	# we save the logits for visualization later
	all_logits.append(logits.detach())

	criterion = nn.MSELoss()


	loss = criterion(logits[train_mask, :], labels[train_mask, :])

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	val_loss = criterion(logits[val_mask, :], labels[val_mask, :])

	if epoch >= 3:
		dur.append(time.time() - t0)

	print('Epoch %d | Loss: %.4f | Validation Loss: %.4f | Time(s) %.4f' % (epoch, loss.item(), val_loss.item(), np.mean(dur)))

print('Finished training!')

test_loss = criterion(logits[test_mask, :], labels[test_mask, :])

print(f'Test loss {test_loss}')


