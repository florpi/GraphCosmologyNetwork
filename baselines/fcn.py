import h5py
import pickle
from GNN.inputs import split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F

# **************************** DEFINE HYPERPARAMS ***********************
learning_rate = 1.e-3
num_epochs = 100
hidden_size = 512 
loss_dict = {
		'learning_rate': learning_rate,
		'hidden_size': conv_hidden_size,
		}



lc_path = '../outputs/learning_curves/'
hdf5_filename = '/cosma5/data/dp004/dc-cues1/features/halo_features_s99'
with h5py.File(hdf5_filename,'r+') as feats: 
	
	features = np.column_stack([feats['M200c'][:], 
							feats['R200c'][:],
							feats['N_subhalos'][:],
							feats['VelDisp'][:],
							feats['Vmax'][:],
							feats['Spin'][:],
							feats['Fsub'][:],
							feats['x_offset'][:]])


	labels = feats['Ngals'][:]

n_features = features.shape[-1]
train_idx, test_idx, val_idx = split.train_test_val_split(labels.shape[0])

train_mask = torch.Tensor(train_idx).long()
test_mask = torch.Tensor(test_idx).long()
val_mask = torch.Tensor(val_idx).long()

train_features = features[train_idx,:]
train_labels = labels[train_idx] 
val_features = features[val_idx,:]
val_labels = labels[val_idx]
test_features = features[test_idx,:]
test_labels = labels[test_idx]

scaler = StandardScaler()

scaler.fit(train_features)

# Standarize features based on training set statistics
std_features = scaler.transform(features)

std_features = torch.tensor(std_features).float()
labels = torch.tensor(labels).float().unsqueeze(-1)
class Net(nn.Module):
		def __init__(self ):
				super(Net, self).__init__()
				self.fc1 = nn.Linear(n_features, hidden_size) 
				self.fc2 = nn.Linear(hidden_size, hidden_size)
				self.fc3 = nn.Linear(hidden_size, 1)


		def forward(self, x):
				x = F.relu(self.fc1(x))
				x = F.relu(self.fc2(x))
				x = self.fc3(x)
				return x


net = Net()


train_loss, validation_loss = [], []
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
for epoch in range(num_epochs):

	logits = net(std_features)

	criterion = nn.MSELoss()


	loss = criterion(logits[train_mask, :], labels[train_mask, :])
	train_loss.append(loss.item())

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	val_loss = criterion(logits[val_mask, :], labels[val_mask, :])
	validation_loss.append(val_loss.item())


	print('Epoch %d | Loss: %.4f | Validation Loss: %.4f ' % (epoch, loss.item(), val_loss.item()))

print('Finished training!')

test_loss = criterion(logits[test_mask, :], labels[test_mask, :])

print(f'Test loss {test_loss}')

loss_dict = {'train': train_loss, 'val': validation_loss}
with open(lc_path + 'fcn.pickle', 'wb') as handle:
	    pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

