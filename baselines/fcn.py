import h5py
import pickle
from GNN.inputs import split
from GNN.utils.cm import plot_confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# **************************** DEFINE HYPERPARAMS ***********************
learning_rate = 1.e-3
num_epochs =  100 
hidden_size = 512 
n_classes = 2

loss_dict = {

		'learning_rate': learning_rate,
		'hidden_size': hidden_size,
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

scaler = StandardScaler()

scaler.fit(features[train_idx, : ])

# Standarize features based on training set statistics
std_features = scaler.transform(features)

std_features = torch.tensor(std_features).float()

labels = labels > 0
labels = torch.tensor(labels).long()

class Net(nn.Module):
		def __init__(self ):
				super(Net, self).__init__()
				self.fc1 = nn.Linear(n_features, hidden_size) 
				#self.fc2 = nn.Linear(hidden_size, hidden_size)
				self.fc3 = nn.Linear(hidden_size, n_classes)


		def forward(self, x):
				x = F.relu(self.fc1(x))
				#x = F.relu(self.fc2(x))
				x = self.fc3(x)
				return x


net = Net()


train_loss, validation_loss = [], []
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
for epoch in range(num_epochs):

	logits = net(std_features)

	criterion = nn.CrossEntropyLoss()


	loss = criterion(logits[train_mask, :], labels[train_mask, ...])
	train_loss.append(loss.item())

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	val_loss = criterion(logits[val_mask, :], labels[val_mask, ...])
	validation_loss.append(val_loss.item())


	print('Epoch %d | Loss: %.4f | Validation Loss: %.4f ' % (epoch, loss.item(), val_loss.item()))

print('Finished training!')

test_loss = criterion(logits[test_mask, :], labels[test_mask, ...])

print(f'Test loss {test_loss}')
test_pred = np.argmax(net(std_features[test_mask, :]).detach().numpy(), axis = -1)
#cm = confusion_matrix(labels[test_mask,...], test_pred) 

#print(cm)
plot_confusion_matrix(labels[test_mask,...], test_pred, classes = ['Dark', 'Luminous'], normalize = True)
plt.savefig('/cosma/home/dp004/dc-cues1/GNN/outputs/cm/fcn.png')

precision, recall, fscore, support = precision_recall_fscore_support(labels[test_mask,...], test_pred) 

print(f'Precision luminuous = {precision[0]:.4f}')
print(f'Precision dark = {precision[1]:.4f}')
print(f'Recall luminuous = {recall[0]:.4f}')
print(f'Recall dark = {recall[1]:.4f}')
print(f'Fscore luminuous = {fscore[0]:.4f}')
print(f'Fscore dark = {fscore[1]:.4f}')



loss_dict["train"] = train_loss
loss_dict["val"] = validation_loss
loss_dict["test"] = test_loss
with open(lc_path + 'fcn.pickle', 'wb') as handle:
	    pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



