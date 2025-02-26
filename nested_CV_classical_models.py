"""Nested cross-validation loop for the classical ML models"""
import os
import numpy as np
import pandas as pd
import joblib
import optuna

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from docs.data_preprocess import *

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("predictions", exist_ok=True)
os.makedirs("best_hyperparameters", exist_ok=True)
os.makedirs("splits", exist_ok=True)

# User inputs
path_to_database = 'datasets/d3tales_public.csv'
target_name = 'solv_oxidation_potential'
desc_sets_filename = ['desc_2D_all.csv', 'desc_MFP_all.csv', 'desc_2D_3D_all.csv']

# Step 1: Calculate and save the descriptor sets
db = read_db(path_to_database, name_target=target_name, smiles_col='smiles', debugging=True)
print("Calculating 2D descriptors...")
desc_2D = get_2D_descriptors(db, autocorr=True)
desc_2D[target_name] = db[target_name].loc[desc_2D.index]
desc_2D.to_csv('datasets/desc_2D_all.csv')
print("Calculating Morgan Fingerprints...")
morgan_FP = get_morgan_fp(db, nBits=2048, radius=2)
morgan_FP[target_name] = db[target_name].loc[morgan_FP.index]
morgan_FP.to_csv('datasets/desc_MFP_all.csv')
print("Calculating 3D descriptors...")
desc_3D = get_3D_descriptors(db)
desc_2D_and_3D = pd.concat([desc_2D, desc_3D, db[target_name]], axis=1, join='inner')
desc_2D_and_3D.to_csv('datasets/desc_2D_3D_all.csv')

del desc_2D, morgan_FP, desc_2D_and_3D

# Define objective function for Optuna
def objective(trial, model_name, X_train, y_train):
    if model_name == "SVR":
        C = trial.suggest_float("C", 1e-2, 1e2, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-3, 1e-1, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        model = SVR(C=C, epsilon=epsilon, gamma=gamma)
    
    elif model_name == "RandomForestRegressor":
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 5, 50)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            n_jobs=-1, random_state=42
        )

    elif model_name == "KNeighborsRegressor":
        n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        p = trial.suggest_int("p", 1, 2)
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)
    
    elif model_name == "HistGradientBoostingRegressor":
        learning_rate = trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        model = HistGradientBoostingRegressor(
            learning_rate=learning_rate, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            random_state=42
        )

    elif model_name == "MLPRegressor":
        hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(50,), (100, 50,), (100, 50, 10,)])
        alpha = trial.suggest_loguniform("alpha", 1e-4, 1e-1)
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-3, 1e-1, log=True)
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
            learning_rate_init=learning_rate_init, early_stopping=True,
            validation_fraction=0.2, max_iter=500, random_state=42
        )

    # Perform 5-fold cross-validation
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    
    return np.mean(score)

# Nested CV process
for split in range(10):
    train_indices, test_indices = train_test_split(db.index, test_size=0.2, random_state=split)
    np.save(f'splits/train_ids_split{split+1}', train_indices)
    np.save(f'splits/test_ids_split{split+1}', test_indices)
    print(f"Saved indices of split {split+1}")

    for desc_set in desc_sets_filename:
        dataset_name = os.path.basename(desc_set).replace("_all.csv", "")
        print(f"\nUsing dataset: {dataset_name}")

        df = pd.read_csv(f'datasets/{desc_set}', index_col=0)
        train_df = df.loc[train_indices]
        test_df = df.loc[test_indices]
        X_train = train_df.drop(columns=[target_name])
        X_test = test_df.drop(columns=[target_name])

        #Preprocess datasets
        train_clean, test_clean = outlier_removal_IF(X_train, X_test, contamination=0.1)
        train_scaled, test_scaled = standard_scaling(train_clean, test_clean)
        train_reduced, test_reduced = feature_reduction(train_scaled, test_scaled,
                                                          var_threshold=0.05, corr_threshold=0.85)
        
        X_train, X_test = train_reduced.copy(), test_reduced.copy()
        y_train = train_df[target_name].loc[X_train.index]
        y_test = test_df[target_name].loc[X_test.index]

        #Optimize and train models
        predictions_df = pd.DataFrame(index=pd.Index(list(X_train.index) + list(X_test.index)))

        for model_name in ["SVR", "RandomForestRegressor", "KNeighborsRegressor", "HistGradientBoostingRegressor", "MLPRegressor"]:
            print(f'\nOptimizing and training: {model_name}')

            #Hyperparameter tuning with optuna
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, model_name, X_train, y_train), n_trials=50, n_jobs=-1)

            best_params = study.best_params
            print(f"Best params for {model_name}: {best_params}")

            # Save best hyperparameters
            with open(f"best_hyperparameters/best_params_{split+1}.txt", "a") as f:
                f.write(f"Best Parameters for {model_name} in dataset {dataset_name}: {best_params}\n")
                f.write("-" * 40 + "\n")

            # Train final model with best hyperparameters
            if model_name == "SVR":
                final_model =  SVR(**best_params)
            elif model_name == "RandomForestRegressor":
                final_model = RandomForestRegressor(**best_params, n_jobs=-1, random_state=42)
            elif model_name == "KNeighborsRegressor":
                final_model = KNeighborsRegressor(**best_params)
            elif model_name == "HistGradientBoostingRegressor":
                final_model = HistGradientBoostingRegressor(**best_params, random_state=42)
            elif model_name == "MLPRegressor":
                final_model = MLPRegressor(**best_params, early_stopping=True, validation_fraction=0.2, max_iter=500, random_state=42)


            final_model.fit(X_train, y_train)
            model_filename = f"models/split{split+1}_{dataset_name}_{model_name}_best.pkl"
            joblib.dump(final_model, model_filename)
            print(f"Model saved: {model_filename}")

            # Predict on train and test sets
            y_train_pred = final_model.predict(X_train)
            y_test_pred = final_model.predict(X_test)

            # Compute metrics
            r2_train = r2_score(y_train, y_train_pred)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            mse_train = mean_squared_error(y_train, y_train_pred)

            r2_test = r2_score(y_test, y_test_pred)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)

            # Write metrics to a txt file
            with open(f"predictions/metrics_split{split+1}.txt", 'a') as f:
                f.write(f"Metrics for desc: {dataset_name} and model: {model_name}\n")
                f.write(f"Train Metrics: R2: {r2_train:.4f} | MAE: {mae_train:.4f} | MSE: {mse_train:.4f}\n")
                f.write(f"Test Metrics: R2: {r2_test:.4f} | MAE: {mae_test:.4f} | MSE: {mse_test:.4f}\n")
                f.write("-" * 40 + "\n")

            # Store predictions
            predictions_df[f'y_pred_{model_name}'] = np.concatenate((y_train_pred, y_test_pred), axis=None)

        # Save predictions
        X_train['split'] = 'train'
        X_test['split'] = 'test'
        desc_df = pd.concat([X_train, X_test], axis=0)
        y_df = pd.concat([y_train, y_test], axis=0)
        final_df = pd.concat([desc_df, y_df, predictions_df], axis=1, join='inner')

        results_filename = f"predictions/split{split+1}_{dataset_name}_best_predictions.csv"
        final_df.to_csv(results_filename)
        print(f"Predictions saved: {results_filename}")
