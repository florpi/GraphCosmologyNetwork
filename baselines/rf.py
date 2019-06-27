import h5py
from GNN.inputs import split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score


hdf5_filename = "/cosma5/data/dp004/dc-cues1/features/halo_features_s99"
with h5py.File(hdf5_filename, "r+") as feats:
    print(feats["boxsize"].value)

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
    

    labels = np.log10(feats["stellar_mass"][:] + 1 )

'''
train_idx, test_idx, val_idx = split.train_test_val_split(labels.shape[0])

train_features = np.concatenate((features[train_idx, :], features[val_idx, :]))
train_labels = np.concatenate((labels[train_idx], labels[val_idx]))
test_features = features[test_idx, :]
test_labels = labels[test_idx]

scaler = StandardScaler()

scaler.fit(train_features)

# Standarize features based on training set statistics
std_features_train = scaler.transform(train_features)
std_features_test = scaler.transform(test_features)


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=5)]
# Number of features to consider at every split
max_features = ["auto", "sqrt"]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 80, num=5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
}

print(random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=100,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=64,
)

# Fit the random search model
rf_random.fit(std_features_train, train_labels)


best_random = rf_random.best_estimator_

random_mse = mean_squared_error(test_labels, best_random.predict(std_features_test))
random_rs2 = r2_score(test_labels, best_random.predict(std_features_test))


print(f"Best random search found a test mse of {random_mse}")
print(f"Best random search found a test rs2 of {random_rs2}")
'''
