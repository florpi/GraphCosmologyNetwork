import os
import time
from datetime import datetime
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Callable

from GNN.inputs.datautils import _train_test_val_split as train_test_val_split
from GNN.inputs import generate_graph
from GNN.models import gcn, gat
from GNN.utils.cm import plot_confusion_matrix
from GNN.utils.checkpointing import CheckpointManager
from GNN.utils.config import load_config
from GNN.utils.importing import get_class_by_name
from GNN.utils.training import AverageMeter, get_log_dir, update_lr

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    confusion_matrix,
    precision_recall_fscore_support,
)
import h5py

import matplotlib.pyplot as plt

import logging

logging.basicConfig(
    filename="logs/train_%s.log" % datetime.now().strftime("%H%M_%d%m%Y"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

# TODO: print validation loss
#      fix tensorboard
# 	   support for graph net together w fully
# 	   final outputs to evaluate model

# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------


def get_arguments() -> argparse.Namespace:
    """
	Set up an ArgumentParser to get the command line arguments.

	Returns:
		A Namespace object containing all the command line arguments
		for the script.
	"""

    # Set up parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        metavar="N",
        help="Size of the mini-batches during training. " "Default: 64.",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="Total number of training epochs. Default: 100.",
    )
    parser.add_argument(
        "--experiment",
        default="default",
        type=str,
        metavar="PATH",
        help="Name of the experiment to run (must be a folder "
        'in the experiments dir). Default: "default".',
    )
    parser.add_argument(
        "--maximum_distance",
        default=0.1,
        type=float,
        metavar="MD",
        help="Maximum distance to connect nodes when building graph. Default: 0.1 Mpc/h",
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-4,
        type=float,
        metavar="LR",
        help="Initial learning rate. Default: 1e-4.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="Path to checkpoint to be used when resuming " "training. Default: None.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=True,
        help="Use TensorBoard to log training progress? " "Default: True.",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=True,
        help="Train on GPU, if available? Default: True.",
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="Number of workers for DataLoaders. Default: 4.",
    )
    # TODO: implement label selection
    # parser.add_argument(
    # 	"--label",
    # 	action="store_true",
    # 	default="dark_or_light",
    # 	help="What to learn: dark_or_light, nr_of_galaxies, central_or_satellite, ...",
    # )

    # Parse and return the arguments (as a Namespace object)
    arguments = parser.parse_args()
    return arguments


def get_data(hdf5_filename: str):

    with h5py.File(hdf5_filename, "r+") as feats:

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

        labels = feats["Ngals"][:]

    train_idx, test_idx, val_idx = train_test_val_split(labels.shape[0])

    train_mask = torch.Tensor(train_idx).long()
    test_mask = torch.Tensor(test_idx).long()
    val_mask = torch.Tensor(val_idx).long()

    scaler = StandardScaler()

    scaler.fit(features[train_idx, :])

    # Standarize features based on training set statistics
    std_features = scaler.transform(features)

    std_features = torch.tensor(std_features).float()

    labels = labels > 0
    labels = torch.tensor(labels).long()

    return train_mask, test_mask, val_mask, std_features, labels


def train_validate(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    model: torch.nn.Module,
    loss_func: Callable,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
):
    """
	Train the given model for a single epoch using the given dataloader.

	Args:
		dataloader: The dataloader containing the training data.
		model: Instance of the model that is being trained.
		loss_func: A loss function to compute the error between the
			actual and the desired output of the model.
		optimizer: An instance of an optimizer that is used to compute
			and perform the updates to the weights of the network.
		epoch: The current training epoch.
		args: Namespace object containing some global variable (e.g.,
			command line arguments, such as the batch size)
	"""

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # Activate training mode
    model.train()

    # Fetch data and move to device
    features, labels = features.to(args.device), labels.to(args.device)
    labels = labels.squeeze()

    # Clear gradients
    optimizer.zero_grad()

    output = model.forward(features).squeeze()

    loss = loss_func(output[train_mask, :], labels[train_mask, ...])

    # Back-propagate the loss and update the weights
    loss.backward()
    optimizer.step(closure=None)

    # ---------------------------------------------------------------------
    # Log information about current batch to TensorBoard
    # ---------------------------------------------------------------------

    if args.tensorboard:
        # Compute how many examples we have processed already and log the
        # loss value for the current batch
        # global_step = ((epoch - 1) * args.n_train_batches + batch_idx) * \
        # 			  args.batch_size
        args.logger.add_scalar(
            tag="loss/train", scalar_value=loss.item(), global_step=epoch
        )

    # ---------------------------------------------------------------------
    # Additional logging to console
    # ---------------------------------------------------------------------

    # Store the loss and processing time for the current batch

    # Print information to console, if applicable

    # Print some information about how the training is going
    logging.info(f"Epoch: {epoch:>3}/{args.epochs}", end=" | ", flush=True)
    logging.info(f"Loss: {loss.item():.6f}", end=" | ", flush=True)

    # Activate model evaluation mode
    model.eval()

    val_loss = loss_func(output[val_mask, :], labels[val_mask, ...])

    return val_loss


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # Start the stopwatch
    script_start = time.time()

    # Read in command line arguments
    args = get_arguments()

    logging.info("")
    logging.info(f"TRAINING NETWORK")
    logging.info("")

    logging.info("Preparing the training process:")
    logging.info(80 * "-")

    # -------------------------------------------------------------------------
    # Load the experiment configuration
    # -------------------------------------------------------------------------

    # Construct the path to the experiment config file
    experiment_dir = os.path.join("experiments", args.experiment)
    config_file_path = os.path.join(experiment_dir, "config.json")

    # Load the config
    config = load_config(config_file_path=config_file_path)

    # -------------------------------------------------------------------------
    # Set up CUDA for GPU support
    # -------------------------------------------------------------------------

    if torch.cuda.is_available() and args.use_cuda:
        args.device = "cuda"
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logging.info(f"device: \t\t GPU ({device_count} x {device_name})")
    else:
        args.device = "cpu"
        logging.info("device: \t\t CPU [CUDA not requested or unavailable]")

    # -------------------------------------------------------------------------
    # Set up the network model
    # -------------------------------------------------------------------------

    # Create a new instance of the model we want to train (specified in the
    # experiment config file), using the desired model parameters
    model_class = get_class_by_name(
        module_name=config["model"]["module"], class_name=config["model"]["class"]
    )
    model = model_class(**config["model"]["parameters"])

    logging.info("model: \t\t\t", model.__class__.__name__)

    # DataParallel will divide and allocate batch_size to all available GPUs
    if args.device == "cuda":
        model = torch.nn.DataParallel(model)

    # Move model to the correct device
    model.to(args.device)

    # -------------------------------------------------------------------------
    # Instantiate an optimizer, a loss function and a LR scheduler
    # -------------------------------------------------------------------------

    # Instantiate the specified optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.learning_rate, amsgrad=True
    )
    logging.info("optimizer: \t\t", optimizer.__class__.__name__)

    # Define the loss function (we use a simple MSE loss)
    loss_func = torch.nn.CrossEntropyLoss().to(args.device)
    logging.info("loss_function: \t\t", loss_func.__class__.__name__)

    # Reduce the LR by a factor of 0.5 if the validation loss did not
    # go down for at least 10 training epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=0.5, patience=8, min_lr=1e-6
    )

    # -------------------------------------------------------------------------
    # Instantiate a CheckpointManager and load checkpoint (if desired)
    # -------------------------------------------------------------------------

    # Construct path to checkpoints directory
    chkpt_dir = os.path.join(experiment_dir, "checkpoints")
    Path(chkpt_dir).mkdir(exist_ok=True)

    # Instantiate a new CheckpointManager
    checkpoint_manager = CheckpointManager(
        model=model,
        checkpoints_directory=chkpt_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        mode="min",
        step_size=-1,
    )

    # Check if we are resuming training, and if so, load the checkpoint
    if args.resume is not None:

        # Load the checkpoint from the provided checkpoint file
        checkpoint_manager.load_checkpoint(args.resume)
        args.start_epoch = checkpoint_manager.last_epoch + 1

        # Print which checkpoint we are using and where we start to train
        logging.info(
            f"checkpoint:\t\t {args.resume} "
            f"(epoch: {checkpoint_manager.last_epoch})"
        )

    # Other, simply print that we're not using any checkpoint
    else:
        args.start_epoch = 1
        logging.info("checkpoint: \t\t None")

    # -------------------------------------------------------------------------
    # Load datasets for training and validation
    # -------------------------------------------------------------------------

    hdf5_filename = "/cosma5/data/dp004/dc-cues1/features/halo_features_s99"

    train_mask, test_mask, val_mask, std_features, labels = get_data(hdf5_filename)

    # -------------------------------------------------------------------------
    # Create a TensorBoard logger and log some basics
    # -------------------------------------------------------------------------

    if args.tensorboard:

        # Create a dir where all the TensorBoard logs will be stored
        tensorboard_dir = os.path.join(experiment_dir, "tensorboard")
        Path(tensorboard_dir).mkdir(exist_ok=True)

        # Create TensorBoard logger
        args.logger = SummaryWriter(log_dir=get_log_dir(log_base_dir=tensorboard_dir))

        # Add all args to as text objects (to epoch 0)
        for key, value in dict(vars(args)).items():
            args.logger.add_text(tag=key, text_string=str(value), global_step=0)

    # -------------------------------------------------------------------------
    # Train the network for the given number of epochs
    # -------------------------------------------------------------------------

    logging.info(80 * "-" + "\n\n" + "Training the model:\n" + 80 * "-")

    for epoch in range(args.start_epoch, args.epochs):

        print("")
        epoch_start = time.time()

        # ---------------------------------------------------------------------
        # Train the model for one epoch
        # ---------------------------------------------------------------------

        validation_loss = train_validate(
            features=std_features,
            labels=labels,
            train_idx=train_mask,
            val_idx=val_mask,
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
        )

        # ---------------------------------------------------------------------
        # Take a step with the CheckpointManager
        # ---------------------------------------------------------------------

        # This will create checkpoint if the current model is the best we've
        # seen yet, and also once every `step_size` number of epochs.
        checkpoint_manager.step(metric=validation_loss, epoch=epoch)

        # ---------------------------------------------------------------------
        # Update the learning rate of the optimizer (using the LR scheduler)
        # ---------------------------------------------------------------------

        # Take a step with the LR scheduler; print message when LR changes
        current_lr = update_lr(scheduler, optimizer, validation_loss)

        # Log the current value of the LR to TensorBoard
        if args.tensorboard:
            args.logger.add_scalar(
                tag="learning_rate", scalar_value=current_lr, global_step=epoch
            )

        # ---------------------------------------------------------------------
        # Print epoch duration
        # ---------------------------------------------------------------------

        print(f"Total Epoch Time: {time.time() - epoch_start:.3f}s\n")

        # ---------------------------------------------------------------------

    print(80 * "-" + "\n\n" + "Training complete!")
    """

	val_prediction, val_label = predict(dataloader = validation_dataloader,
							model = model,
							num_outputs = 200,
							args = args)


	# -------------------------------------------------------------------------
	# Postliminaries
	# -------------------------------------------------------------------------

	print('')
	print(f'This took {time.time() - script_start:.1f} seconds!')
	print('')



# **************************** DEFINE HYPERPARAMS ***********************
readout_hidden_size = 512
n_classes = 2
model = "gcn"

# **************************** INPUT/ OUTPUT DIRSi ***********************
lc_path = "outputs/learning_curves/"
hdf5_filename = "/cosma5/data/dp004/dc-cues1/features/halo_features_s99"


# **************************** DEFINE GRAPH ***********************
# TODO: In depth exploration input graph
labels, G = generate_graph.hdf52graph(hdf5_filename, maximum_distance)

num_features = G.ndata["feat"].shape[-1]


#embedding_size = conv_hidden_size + num_features
conv_hidden_size = num_features
embedding_size = conv_hidden_size

loss_dict = {
	"maximum_distance": maximum_distance,
	"learning_rate": learning_rate,
	"conv_hidden_size": conv_hidden_size,
	"readout_hidden_size": readout_hidden_size,
	"model": model,
}


# ****************** MASKS FOR TRAIN/VAL/SPLIT ***********************

train_idx, test_idx, val_idx = split.train_test_val_split(len(G.nodes()))

train_mask = torch.Tensor(train_idx).long()
test_mask = torch.Tensor(test_idx).long()
val_mask = torch.Tensor(val_idx).long()

scaler = StandardScaler()

scaler.fit(G.ndata["feat"][train_idx, :])

# Standarize features based on training set statistics
G.ndata["std_feat"] = torch.Tensor(scaler.transform(G.ndata["feat"])).float()


# ******************  DEFINE NETWORK ***********************


if model == "gat":
	net = gat.GAT(
		G,
		in_dim=G.ndata["std_feat"].shape[-1],
		hidden_dim=conv_hidden_size,
		embedding_size=embedding_size,
		readout_hidden_size=readout_hidden_size,
		num_heads=2,
	)

else:
	net = gcn.GCN(
		G.ndata["std_feat"].shape[-1],
		conv_hidden_size,
		embedding_size,
		readout_hidden_size,
		num_classes = n_classes
	)

# ******************  TRAINING LOOP ***********************


print(net.parameters())

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
all_logits, dur, train_loss, validation_loss = [], [], [], []

for epoch in range(num_epochs):

	if epoch >= 3:
		t0 = time.time()

	if model == "gat":
		logits = net(torch.tensor(G.ndata["std_feat"]).float())

	else:
		logits = net(G, torch.tensor(G.ndata["std_feat"]).float())

	# we save the logits for visualization later
	all_logits.append(logits.detach())

	criterion = nn.CrossEntropyLoss()
	#criterion = nn.MSELoss()

	loss = criterion(logits[train_mask, :], labels[train_mask, ...])
	train_loss.append(loss.item())

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	val_loss = criterion(logits[val_mask, :], labels[val_mask, ...])
	validation_loss.append(val_loss.item())

	if epoch >= 3:
		dur.append(time.time() - t0)

	print(
		"Epoch %d | Loss: %.4f | Validation Loss: %.4f | Time(s) %.4f"
		% (epoch, loss.item(), val_loss.item(), np.mean(dur))
	)

print("Finished training!")

test_loss = criterion(logits[test_mask, :], labels[test_mask, ...])

print(f"Test loss {test_loss}")
test_pred = np.argmax(net(G, torch.tensor(G.ndata["std_feat"]).float())[test_mask, :].detach().numpy(), axis = -1)
#cm = confusion_matrix(labels[test_mask,...], test_pred)

#print(cm)
plot_confusion_matrix(labels[test_mask,...], test_pred, classes = ['Dark', 'Luminous'], normalize = True)
plt.savefig(f'/cosma/home/dp004/dc-cues1/GNN/outputs/cm/{model}_d{maximum_distance}.png')


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
with open(lc_path + f"{model}_d{maximum_distance}_sigmoid.pickle", "wb") as handle:
	pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
