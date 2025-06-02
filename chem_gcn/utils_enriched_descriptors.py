"""
Script modified from: 
    G. S. Deshmukh, “ChemGCN: A Graph Convolutional Network for Chemical Property
    Prediction,” GitHub repository, 2020. Available: https://github.com/gauravsdeshmukh/ChemGCN
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score

class Standardizer:
    def __init__(self, x):
        self.mean = torch.mean(x)
        self.std = torch.std(x)

    def standardize(self, x):
        Z = (x-self.mean)/(self.std)
        Z = Z.view(-1) ##OUTPUT TENSOR
        return Z
    
    def restore(self, Z):
        x = self.mean + Z*self.std
        return x

    def state(self):
        return {"mean": self.mean, "std": self.std}

    def load(self, state):
        self.mean = state["mean"]
        self.std = state["std"]

# Utility functions to train, test model
def train_model(
    epoch,
    model,
    training_dataloader,
    optimizer,
    loss_fn,
    standardizer,
    device,
    max_atoms,
    node_vec_len,
    descriptors_len,
):
    """
    Execute training of one epoch for the ChemGCN model.

    Parameters
    ----------
    epoch : int
        Current epoch
    model : ChemGCN
        ChemGCN model object
    training_dataloader : data.DataLoader
        Training DataLoader
    optimizer : torch.optim.Optimizer
        Model optimizer
    loss_fn : like nn.MSELoss()
        Model loss function
    standardizer : Standardizer
        Standardizer object
    use_GPU: bool
        Whether to use GPU
    max_atoms: int
        Maximum number of atoms in graph
    node_vec_len: int
        Maximum node vector length in graph
    descriptor_len: int
        Number of molecular descriptors to concatenate

    Returns
    -------
    avg_loss : float
        Training loss averaged over batches
    avg_mae : float
        Training MAE averaged over batches
    """

    # Create variables to store losses and error
    avg_loss = 0
    avg_mae = 0
    count = 0

    # Switch model to train mode
    model.train()

    # Go over each batch in the dataloader
    for i, dataset in enumerate(training_dataloader):
        # Unpack data
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        output = dataset[1]
        ##ADDED to get the molecular descriptors
        descriptors = dataset[3]
        ids = dataset[4]

        # Reshape inputs
        first_dim = int((torch.numel(node_mat)) / (max_atoms * node_vec_len)) 
        node_mat = node_mat.reshape(first_dim, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(first_dim, max_atoms, max_atoms)
        descriptors = descriptors.reshape(first_dim, descriptors_len) ##CHANGE

        # Standardize output
        output_std = standardizer.standardize(output)

        # Package inputs and outputs; check if GPU is enabled
        nn_input = (node_mat.to(device), adj_mat.to(device), descriptors.to(device))
        nn_output = output_std.to(device)

        # Compute output from network
        nn_prediction, neural_fingerprints = model(*nn_input)
        nn_prediction = torch.squeeze(nn_prediction, dim=1) #ADDED BY ME

        # Calculate loss
        loss = loss_fn(nn_output, nn_prediction)
        avg_loss += loss

        # Calculate MAE
        prediction = standardizer.restore(nn_prediction.detach().cpu())
        mae = mean_absolute_error(output, prediction)
        avg_mae += mae

        # Set zero gradients for all tensors
        optimizer.zero_grad()

        # Do backward prop
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Set your desired max_norm value

        # Update optimizer parameters
        optimizer.step()

        # Increase count
        count += 1

    # Calculate avg loss and MAE
    avg_loss = avg_loss.detach().cpu().numpy() / count
    avg_mae = avg_mae / count

    # Print stats
    print(
        "Epoch: [{0}]\tTraining Loss: [{1:.2f}]\tTraining MAE: [{2:.2f}]".format(
            epoch, avg_loss, avg_mae
        )
    )

    # Return loss and MAE
    return avg_loss, avg_mae



def predict_gcn(model,test_dataloader,
                standardizer,device,max_atoms,node_vec_len,
                descriptors_len,):
    """
    Predict and return predictions with IDs

    """

    # Create variables to store losses and error
    outputs = []
    predictions = []
    ids = []
    neural_fingerprints = []

    # Switch model to train mode
    model.eval()

    # Go over each batch in the dataloader
    for i, dataset in enumerate(test_dataloader):
        # Unpack data
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        output = dataset[1]
        descriptors = dataset[3]
        id = dataset[4]

        # Reshape inputs
        first_dim = int((torch.numel(node_mat)) / (max_atoms * node_vec_len))
        node_mat = node_mat.reshape(first_dim, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(first_dim, max_atoms, max_atoms)
        descriptors = descriptors.reshape(first_dim, descriptors_len) #CHANGE DIMENSION ACCORDINGLY


        # Package inputs and outputs; check if GPU is enabled
        nn_input = (node_mat.to(device), adj_mat.to(device), descriptors.to(device))
        
        # Compute output from network
        nn_prediction, neural_fingerprint = model(*nn_input)
        nn_prediction = torch.squeeze(nn_prediction, dim=1)

        # Calculate MAE
        prediction = standardizer.restore(nn_prediction.detach().cpu())

        # Add to list
        outputs.append(output)
        predictions.append(prediction)
        ids.append(id)
        neural_fingerprints.append(neural_fingerprint)

    # Flatten
    preds_arr = np.concatenate(predictions)
    ids_arr = np.concatenate(ids)
    neural_fingerprints_arr = np.concatenate([tensor.cpu().numpy() for tensor in neural_fingerprints])

    return preds_arr, ids_arr, neural_fingerprints_arr


def loss_curve(save_dir, epochs, losses):
    """
    Make a loss curve.

    Parameters
    ----------
    save_dir: str
        Name of directory to store plot in
    epochs: list
        List of epochs
    losses: list
        List of losses

    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=500)
    ax.plot(epochs, losses, marker="o", linestyle="--", color="royalblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared loss")
    ax.set_title("Loss curve")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_curve.png"))
    