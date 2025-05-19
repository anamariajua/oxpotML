import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import optuna  
from optuna.pruners import MedianPruner  
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

#### Start by creating dataset
try:
    main_path = Path(__file__).resolve().parent
except NameError:
    main_path = Path(os.getcwd()).resolve()

data_path = main_path / "predictions/split10_desc_2D_3D_best_predictions.csv"
df = pd.read_csv(data_path, index_col=0)
df = df[['split', 'solv_oxidation_potential']]

df_smiles = pd.read_csv('datasets/d3tales_clean.csv', index_col=0)
df_all = pd.concat([df, df_smiles['smiles']], axis=1, join='inner')

dataset = GraphData(
    dataset_df=df_all,
    max_atoms=max_atoms,
    node_vec_len=atomic_features['node_vec_len'],
    atom_types_dict=atomic_features['atom_types'],
    aromaticity_dict=atomic_features['aromaticity'],
    hybridization_dict=atomic_features['hybridization'],
    formal_charge_dict=atomic_features['formal_charge'],
    ea_dict=atomic_features['ea'],
    ip_dict=atomic_features['ip'],
    atomic_mass_dict=atomic_features['atomic_mass'],
    total_valence_e_dict=atomic_features['total_valence_e'],
    en_dict=atomic_features['en'],
    atomic_radius_dict=atomic_features['atomic_radius'],
)
print("Finished creating dataset (GraphData) object")

# Define objective function for Optuna
def objective(trial, train_id, dataset_object, split):
    hidden_nodes = trial.suggest_int("hidden_nodes", 50, 200)
    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 4)
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 2, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
    p_dropout = trial.suggest_float("p_dropout", 0.1, 0.5)

    gpu_id = trial.number % NUM_GPUS  
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Partition train-validation indices in 3-fold CV 
    kf = KFold(n_splits=3)
    train_val_indices = train_id
    scores = []

    for l, (train_pos_indices, test_pos_indices) in enumerate(kf.split(train_val_indices)):
        # Create dataloaders
        train_indices = train_val_indices[train_pos_indices]
        test_indices = train_val_indices[test_pos_indices]
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_loader = DataLoader(
            dataset_object,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collate_graph_dataset
        )
        test_loader = DataLoader(
            dataset_object,
            batch_size=batch_size,
            sampler=test_sampler,
            collate_fn=collate_graph_dataset
        )

        # Initialize model
        model = ChemGCN(
            node_vec_len=atomic_features['node_vec_len'],
            node_fea_len=hidden_nodes,
            hidden_fea_len=hidden_nodes,
            n_conv=n_conv_layers,
            n_hidden=n_hidden_layers,
            n_outputs=1,
            p_dropout=p_dropout
        ).to(device)

        # Standardizer
        outputs_train = [
            dataset_object.outputs.get(i)
            for i in train_indices
            if dataset_object.outputs.get(i) is not None
        ]
        standardizer = Standardizer(torch.Tensor(outputs_train))

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Loss function
        loss_fn = nn.MSELoss()

        # Train
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

        # Evaluate on this fold
        preds_test, ids_test, _ = predict_gcn(
            model=model,
            test_dataloader=test_loader,
            standardizer=standardizer,
            device=device,
            max_atoms=max_atoms,
            node_vec_len=atomic_features['node_vec_len']
        )
        outputs_test = [dataset_object.outputs.get(i) for i in ids_test]
        fold_mse_test = mean_squared_error(outputs_test, preds_test)
        fold_score = -fold_mse_test  # negative MSE => "maximize"
        scores.append(fold_score)

        partial_avg_score = np.mean(scores)  # average negative MSE so far
        trial.report(partial_avg_score, step=l)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return the average across folds
    return np.mean(scores)


# Nested CV process
predictions_df = pd.DataFrame()

train_indices = np.load(f'splits/train_ids_split10.npy', allow_pickle=True)
test_indices = np.load(f'splits/test_ids_split10.npy', allow_pickle=True)

print(f"Reading indices of split 10")

dataset_indices_set = set(dataset.indices)
train_set = set(train_indices)
test_set = set(test_indices)

missing_train = train_set - dataset_indices_set
missing_test = test_set - dataset_indices_set

print("The size of train_indices:", len(train_indices))
print("The size of test_indices:", len(test_indices))

train_indices = np.array([idx for idx in train_indices if idx in dataset_indices_set])
test_indices = np.array([idx for idx in test_indices if idx in dataset_indices_set])

print("The size of train_indices after filtering:", len(train_indices))
print("The size of test_indices after filtering:", len(test_indices))
print("Index check complete for split")

# [PRUNING CHANGE] Create study with a MedianPruner.
pruner = MedianPruner(n_warmup_steps=1)
study = optuna.create_study(direction="maximize", pruner=pruner)

print(f'\nHyperparameter tuning using standard GCN split 10:')
study.optimize(
    lambda trial: objective(trial, train_indices, dataset, split=None),
    n_trials=50,
    n_jobs=NUM_GPUS  
)

best_params = study.best_params
print(f"Best params for standard GCN are: {best_params}")

# Save best hyperparameters
with open(f"best_hyperparameters/best_params_GCN.txt", "a") as f:
    f.write(f"Best Parameters for standard GCN in split 10: {best_params}\n")
    f.write("-" * 40 + "\n")

# Create dataloaders
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    collate_fn=collate_graph_dataset
)
test_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=test_sampler,
    collate_fn=collate_graph_dataset
)

gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

# Initialize model with best hyperparams
model = ChemGCN(
    node_vec_len=atomic_features['node_vec_len'],
    node_fea_len=best_params.get('hidden_nodes'),
    hidden_fea_len=best_params.get('hidden_nodes'),
    n_conv=best_params.get('n_conv_layers'),
    n_hidden=best_params.get('n_hidden_layers'),
    n_outputs=1,
    p_dropout=best_params.get('p_dropout')
).to(device)

# Standardizer
outputs_train = [dataset.outputs.get(i) for i in train_indices]
standardizer = Standardizer(torch.Tensor(outputs_train))

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=best_params.get('learning_rate'))
loss_fn = nn.MSELoss()

# Final training with best hyperparams
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
        atomic_features['node_vec_len'],
    )

torch.save(model.state_dict(), f"models/standard_GCN_split10.pth")
print("Model saved in models/")

preds_train, ids_train, neural_fp_train = predict_gcn(
    model=model,
    test_dataloader=train_loader,
    standardizer=standardizer,
    device=device,
    max_atoms=max_atoms,
    node_vec_len=atomic_features['node_vec_len']
)
preds_test, ids_test, neural_fp_test = predict_gcn(
    model=model,
    test_dataloader=test_loader,
    standardizer=standardizer,
    device=device,
    max_atoms=max_atoms,
    node_vec_len=atomic_features['node_vec_len']
)

outputs_train = [dataset.outputs.get(i) for i in ids_train]
outputs_test = [dataset.outputs.get(i) for i in ids_test]

df_train = pd.DataFrame({"set": "train", "neural_fp": [neural_fp_train[i] for i in range(len(neural_fp_train))]}, index=ids_train)
df_test = pd.DataFrame({"set": "test", "neural_fp": [neural_fp_test[i] for i in range(len(neural_fp_test))]}, index=ids_test)
df = pd.concat([df_train, df_test])

r_squared_train = r2_score(outputs_train, preds_train)
mae_train = mean_absolute_error(outputs_train, preds_train)
mse_train = mean_squared_error(outputs_train, preds_train)
r_squared_test = r2_score(outputs_test, preds_test)
mae_test = mean_absolute_error(outputs_test, preds_test)
mse_test = mean_squared_error(outputs_test, preds_test)

with open("predictions/metrics_standard_GCN.txt", 'a') as f:
    f.write("Metrics for standard GCN model: split 10\n")
    f.write(f"Train Metrics: R2: {r_squared_train:.4f} | MAE: {mae_train:.4f} | MSE: {mse_train:.4f}\n")
    f.write(f"Test Metrics: R2: {r_squared_test:.4f} | MAE: {mae_test:.4f} | MSE: {mse_test:.4f}\n")
    f.write("-" * 40 + "\n")

y_train_pred = pd.Series(preds_train, index=ids_train)
y_test_pred = pd.Series(preds_test, index=ids_test)
df['y_pred_10'] = pd.concat([y_train_pred, y_test_pred], axis=0)

results_filename = "predictions/standard_GCN_fps_prediction_split10.csv"
df.to_csv(results_filename)
print(f" Predictions saved: {results_filename}")
