import os, time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Callable

import numpy as np

import pickle

from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    precision_recall_fscore_support,
)

from GNN.inputs import get_data, split
from GNN.utils.cm import plot_confusion_matrix
from GNN.utils.checkpointing import CheckpointManager
from GNN.utils.config import load_config
from GNN.utils.importing import get_class_by_name
from GNN.utils.training import AverageMeter, get_log_dir, update_lr

import logging

logging.basicConfig(
    filename="logs/train_%s.log" % datetime.now().strftime("%H%M_%d%m%Y"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

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
    # TODO:
    parser.add_argument(
        "--label",
        action="store_true",
        default="nr_of_galaxies",
        help="What to learn: dark_or_light, nr_of_galaxies, central_or_satellite, ...",
    )
    parser.add_argument(
        "--sampling",
        default="upsampling",
        type=str,
        help="Balance training set via: upsampling, downsampling. Default: upsampling.",
    )
    # TODO:
    parser.add_argument(
        "--sacred",
        action="store_true",
        default=False,
        help="Use Sacred to manage experiments? Default: False.",
    )
    # TODO:
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=False,
        help="Use TensorBoard to log training progress? Default: False.",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        metavar="N",
        help="Size of the mini-batches during training. " "Default: 64.",
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
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="Number of workers for DataLoaders. Default: 4.",
    )

    # Parse and return the arguments (as a Namespace object)
    arguments = parser.parse_args()
    return arguments


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
    # Set up the random-forest model
    # -------------------------------------------------------------------------

    # Create a new instance of the model we want to train (specified in the
    # experiment config file), using the desired model parameters
    model_class = get_class_by_name(
        module_name=config["model"]["module"], class_name=config["model"]["class"]
    )
    model = model_class(**config["model"]["parameters"])

    logging.info("model: \t\t\t", model.__class__.__name__)

    # -------------------------------------------------------------------------
    # Load and prepare datasets
    # -------------------------------------------------------------------------

    # Load dataset
    hdf5_filename = "/cosma5/data/dp004/dc-cues1/features/halo_features_s99"
    feature_names = [
        "M200c",
        "R200c",
        "N_subhalos",
        "VelDisp",
        "Vmax",
        "Spin",
        "Fsub",
        "x_offset",
    ]
    train, test = get_data(hdf5_filename, args.label)

    # Prepare datasets
    ## Balance training set in the transition region
    center_transition = 2.1e11
    end_transition = 8e11

    df_train = pd.DataFrame(dict(zip(feature_names, train["features"].T)))
    df_train["labels"] = train["labels"]

    logging.info(
        "The labels before balancing are as follows:", df_train.labels.value_counts()
    )
    df_train = balance_dataset(
        df_train, center_transition, end_transition, arg.sampling
    )
    logging.info(
        "The labels after balancing are as follows:",
        df_train[df_train.M200c < center_transition].labels.value_counts(),
        df_train[
            (df_train.M200c > center_transition) & (df_train.M200c < end_transition)
        ].labels.value_counts(),
    )

    df_train_features = df_train.drop(columns="labels")
    df_train_labels = df_train["labels"]

    ## Standarize features
    scaler = StandardScaler()
    scaler.fit(df_train_features)
    std_train_features = scaler.transform(df_train_features)
    test_features = scaler.transform(test_features)

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

        logging.info("")
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
