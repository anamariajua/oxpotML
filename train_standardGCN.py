"""Nested cross-validation loop for the standard Graph Convolutional Nueral Network model"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import optuna
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler 
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from chem_gcn.model import ChemGCN
from chem_gcn.utils import (
    train_model,
    predict_gcn,
    Standardizer,
)
from chem_gcn.graphs_new import GraphData, collate_graph_dataset

### Fix seeds
np.random.seed(1)
torch.manual_seed(456)

# Load JSON file
with open("atom_features.json", "r") as file:
    atomic_features = json.load(file)

# Model inputs
max_atoms = 140
train_size = 0.8
batch_size = 64
n_epochs = 50
NUM_GPUS = 4
best_params = {'hidden_nodes': 171, 'n_conv_layers': 4, 
               'n_hidden_layers': 3, 'learning_rate': 0.0015611219566948208, 
               'p_dropout': 0.3482343291857878}

#### Start by creating dataset
# Check if __file__ is defined to decide the method to get the current path
try:
    # If __file__ is defined, use it to get the parent directory of the script
    main_path = Path(__file__).resolve().parent
except NameError:
    # If __file__ is not defined, fallback to the current working directory
    main_path = Path(os.getcwd()).resolve()

# Nested CV process
predictions_df = pd.DataFrame()

for split in range(10):
    data_path = main_path / f"predictions/split{split+1}_desc_2D_3D_best_predictions.csv"
    print(f'Reading dataset: {data_path} \n')
    df = pd.read_csv(data_path, index_col=0)
    df = df.iloc[:,:-5]
    df_smiles = pd.read_csv('datasets/d3tales_clean.csv', index_col=0)
    df_all = pd.concat([df, df_smiles['smiles']], axis=1, join='inner')
    gpu_id = 0 
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    dataset = GraphData(dataset_df=df_all, max_atoms=max_atoms, 
                        node_vec_len=atomic_features['node_vec_len'], 
                        atom_types_dict=atomic_features['atom_types'], aromaticity_dict=atomic_features['aromaticity'], 
                        hybridization_dict=atomic_features['hybridization'], formal_charge_dict=atomic_features['formal_charge'], 
                        ea_dict=atomic_features['ea'], ip_dict=atomic_features['ip'],
                        atomic_mass_dict=atomic_features['atomic_mass'], total_valence_e_dict=atomic_features['total_valence_e'],
                        en_dict=atomic_features['en'], atomic_radius_dict=atomic_features['atomic_radius'])
    print("Finished creating dataset (GraphData) object")

    train_indices = np.load(f'splits/train_ids_split{split+1}.npy', allow_pickle=True)
    test_indices = np.load(f'splits/test_ids_split{split+1}.npy', allow_pickle=True)

    print(f"Reading indices of split {split+1}")

    # Convert dataset indices to a set for fast lookup
    dataset_indices_set = set(dataset.indices)

    # Convert train/test indices to sets
    train_set = set(train_indices)
    test_set = set(test_indices)

    # Check if all train/test indices exist in dataset
    missing_train = train_set - dataset_indices_set  # Train indices not in dataset
    missing_test = test_set - dataset_indices_set  # Test indices not in dataset

    print("The size of train_indices:", len(train_indices))
    print("The size of test_indices:", len(test_indices))

    # Drop missing indices
    train_indices = np.array([idx for idx in train_indices if idx in dataset_indices_set])
    test_indices = np.array([idx for idx in test_indices if idx in dataset_indices_set])

    print("The size of train_indices after filtering:", len(train_indices))
    print("The size of test_indices after filtering:", len(test_indices))

    print("Index check complete for split", split+1)

    # Create dataloaders
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, 
                          sampler=train_sampler, 
                          collate_fn=collate_graph_dataset)
    test_loader = DataLoader(dataset, batch_size=batch_size, 
                         sampler=test_sampler,
                         collate_fn=collate_graph_dataset)

    #### Initialize model, standardizer, optimizer, and loss function:
    # Model
    model = ChemGCN(node_vec_len=atomic_features['node_vec_len'], node_fea_len=best_params.get('hidden_nodes'),
                hidden_fea_len=best_params.get('hidden_nodes'), n_conv=best_params.get('n_conv_layers'), 
                n_hidden=best_params.get('n_hidden_layers'),
                n_outputs=1, p_dropout=best_params.get('p_dropout')).to(device)

    # Standardizer
    outputs_train = [dataset.outputs.get(i) for i in train_indices]
    standardizer = Standardizer(torch.Tensor(outputs_train))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params.get('learning_rate'))

    # Loss function
    loss_fn = torch.nn.MSELoss()

    #### Train the model
    loss = []
    mae = []
    epoch = []
    for i in range(n_epochs):
        epoch_loss, epoch_mae = train_model( 
        i,
        model,
        train_loader,
        optimizer,
        loss_fn,
        standardizer,
        device,
        max_atoms,
        atomic_features['node_vec_len']
    )
        loss.append(epoch_loss)
        mae.append(epoch_mae)
        epoch.append(i)

    torch.save(model.state_dict(), f"models/standard_GCN_split{split+1}.pth")
    print("Model saved in models/")

    preds_train, ids_train, neural_fp_train = predict_gcn(model=model, test_dataloader=train_loader, standardizer=standardizer, 
                                                          device=device, max_atoms=max_atoms, node_vec_len=atomic_features['node_vec_len'],
                                                          )
    preds_test, ids_test, neural_fp_test = predict_gcn(model=model,test_dataloader= test_loader, standardizer=standardizer, 
                                                       device=device, max_atoms=max_atoms, node_vec_len=atomic_features['node_vec_len'],
                                                       )
    outputs_train = [dataset.outputs.get(i) for i in ids_train]
    outputs_test = [dataset.outputs.get(i) for i in ids_test]

    if split == 0:
        df_train = pd.DataFrame({ "set": "train", "neural_fp": [neural_fp_train[i] for i in range(len(neural_fp_train))]}, index=ids_train)
        df_test = pd.DataFrame({"set": "test", "neural_fp": [neural_fp_test[i] for i in range(len(neural_fp_test))]}, index=ids_test)
        # Concatenate both DataFrames
        df = pd.concat([df_train, df_test])
        # Save neural fingerprints
        results_filename = "predictions/neural_fp_standard_GCN.csv"
        df.to_csv(results_filename)
        print(f" Neural Fingerprints saved: {results_filename}")

    # Calculate metrics
    r_squared_train = r2_score(outputs_train, preds_train)
    mae_train = mean_absolute_error(outputs_train, preds_train)
    mse_train = mean_squared_error(outputs_train, preds_train)
    r_squared_test = r2_score(outputs_test, preds_test)
    mae_test = mean_absolute_error(outputs_test, preds_test)
    mse_test = mean_squared_error(outputs_test, preds_test)

    # Write metrics to a txt file
    with open(f"predictions/metrics_standard_GCN.txt", 'a') as f:
        f.write(f"Metrics for standard GCN model: split {split+1}\n")
        f.write(f"Train Metrics: R2: {r_squared_train:.4f} | MAE: {mae_train:.4f} | MSE: {mse_train:.4f}\n")
        f.write(f"Test Metrics: R2: {r_squared_test:.4f} | MAE: {mae_test:.4f} | MSE: {mse_test:.4f}\n")
        f.write("-" * 40 + "\n")

    # Store predictions
    y_train_pred = pd.Series(preds_train, index=ids_train)
    y_test_pred = pd.Series(preds_test, index=ids_test)
    predictions_df[f'y_pred_{split+1}'] = pd.concat([y_train_pred, y_test_pred], axis=0)

# Save predictions
results_filename = f"predictions/standard_GCN_predictions.csv"
predictions_df.to_csv(results_filename)
print(f" Predictions saved: {results_filename}")



