import numpy as np

def train_test_val_split(n_nodes, train_size = 0.7):

	test_val_idx = np.random.choice(n_nodes,
			size = int( (1. - train_size) * n_nodes),
			replace = False)

	val_idx = np.random.choice(test_val_idx, 
			size = int( 0.5 * len(test_val_idx)),
			replace = False)


	test_idx = np.array(list(set(test_val_idx) - set(val_idx)))

	train_idx = np.array(list(set(np.arange(0, n_nodes)) - set(test_val_idx)))


	assert n_nodes == len(val_idx) + len(test_idx) + len(train_idx)

	return train_idx, test_idx, val_idx
