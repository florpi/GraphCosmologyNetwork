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
import pickle


# **************************** DEFINE HYPERPARAMS ***********************
maximum_distance = 1 # Mpc
learning_rate = 1.e-3
num_epochs = 100
conv_hidden_size = 32
embedding_size = conv_hidden_size
readout_hidden_size = 512
model = 'gcn'

loss_dict = {
		'maximum_distance': maximum_distance,
		'learning_rate': learning_rate,
		'conv_hidden_size': conv_hidden_size,
		'readout_hidden_size': readout_hidden_size,
		'model': model
		}

# **************************** INPUT/ OUTPUT DIRSi ***********************
lc_path = 'outputs/learning_curves/'
hdf5_filename = '/cosma5/data/dp004/dc-cues1/features/halo_features_s99'


# **************************** DEFINE GRAPH ***********************
labels, G = generate_graph.hdf52graph(hdf5_filename, maximum_distance)


# ****************** MASKS FOR TRAIN/VAL/SPLIT ***********************

train_idx, test_idx, val_idx = split.train_test_val_split(len(G.nodes()))

train_mask = torch.Tensor(train_idx).long()
test_mask = torch.Tensor(test_idx).long()
val_mask = torch.Tensor(val_idx).long()

scaler = StandardScaler()

scaler.fit(G.ndata['feat'][train_idx,:])

# Standarize features based on training set statistics
G.ndata['std_feat'] = torch.Tensor(scaler.transform(G.ndata['feat'])).float()


# ******************  DEFINE NETWORK ***********************


if model == 'gat':
	net = gat.GAT(G,
		in_dim = G.ndata['std_feat'].shape[-1],
		hidden_dim = conv_hidden_size,
		embedding_size = conv_hidden_size,
		readout_hidden_size = readout_hidden_size,
		num_heads = 2)

else:
	net = gcn.GCN(G.ndata['std_feat'].shape[-1], conv_hidden_size, embedding_size, readout_hidden_size )

# ******************  TRAINING LOOP ***********************


print(net.parameters())

optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
all_logits, dur, train_loss, validation_loss = [], [], [], []

for epoch in range(num_epochs):

	if epoch >= 3:
		t0 = time.time()

	if model == 'gat':
		logits = net(torch.tensor(G.ndata['std_feat']).float())

	else:
		logits = net(G,  torch.tensor(G.ndata['std_feat']).float())

	# we save the logits for visualization later
	all_logits.append(logits.detach())


	criterion = nn.MSELoss()

	loss = criterion(logits[train_mask, :], labels[train_mask, :])
	train_loss.append(loss.item())

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	val_loss = criterion(logits[val_mask, :], labels[val_mask, :])
	validation_loss.append(val_loss.item())

	if epoch >= 3:
		dur.append(time.time() - t0)

	print('Epoch %d | Loss: %.4f | Validation Loss: %.4f | Time(s) %.4f' % (epoch, loss.item(), val_loss.item(), np.mean(dur)))

print('Finished training!')

test_loss = criterion(logits[test_mask, :], labels[test_mask, :])

print(f'Test loss {test_loss}')

loss_dict['train'] = train_loss
loss_dict['val'] = validation_loss
loss_dict['test'] = test_loss 
with open(lc_path + f'{model}.pickle', 'wb') as handle:
	    pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

