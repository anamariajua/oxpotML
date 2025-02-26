"""
This python script includes all necessary functions for the calculation of descriptors and subsequent
preprocessing stage. 
Created by: Ana Maria Juarez
Parts of the code were obtained from the following sources:
-
-
-
"""

import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, Descriptors3D
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold 

def get_2D_descriptors(df, smiles_col='smiles', autocorr = True):
    if autocorr:
        Descriptors.setupAUTOCorrDescriptors()
    smiles = list(df[smiles_col])
    ids = list(df.index)
    desc = [Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(smile)) for smile in smiles]
    df = pd.DataFrame.from_dict(desc).set_index(pd.Index(ids))
    print('There are:', df.isna().sum().sum(axis=0), 'rows with NaN values in dataframe')
    df = df.dropna(axis=0) #Drops rows
    print('Molecules with NaN values were dropped')
    return df

def get_3D_descriptors(df, smiles_col='smiles'):
    smiles = list(df[smiles_col])
    ids = list(df.index)
    array_desc = []
    valid_ids = []  # To keep track of valid molecule indices
    for i, smile in enumerate(smiles):
        lowest_e = find_lowest_e_conf(smile)
        # Skip if lowest_e is None
        if lowest_e is None:
            print(f"Skipping {smile}: No valid conformer found.")
            continue  
        mol = Chem.rdmolfiles.MolFromXYZBlock(lowest_e)
        # Skip if MolFromXYZBlock fails
        if mol is None:
            print(f"Skipping {smile}: Failed to generate molecule from XYZ block.")
            continue  
        desc_3D = Descriptors3D.CalcMolDescriptors3D(mol)
        array_desc.append(desc_3D)
        valid_ids.append(ids[i])  # Keep only valid indices
    # Create DataFrame with only valid molecules
    df_desc = pd.DataFrame.from_dict(array_desc).set_index(pd.Index(valid_ids))
    return df_desc


def get_morgan_fp(df, nBits=2048, radius=2, 
                  smiles_col='smiles'):
    #get list of ids and smiles
    smiles = list(df[smiles_col])
    ids = list(df.index)
    Morgan_FP = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), radius=radius, nBits=nBits) 
                 for smile in smiles]
    ecfp6_name = [f'Bit_{i}' for i in range(nBits)]
    ecfp6_bits = [list(l) for l in Morgan_FP]
    morgan_df = pd.DataFrame(ecfp6_bits, index=pd.Index(ids), columns=ecfp6_name)
    return morgan_df
    

def outlier_removal_IF(train_df, test_df, contamination=0.1, random_state = 42, n_estimators=100, max_samples='auto'):
    print('Performing outlier removal:')
    clf = IsolationForest(random_state=random_state, 
                          n_estimators=n_estimators, 
                          max_samples=max_samples, 
                          contamination=contamination, n_jobs=1).fit(train_df)
    train_outliers = clf.predict(train_df) #Returns -1 for outliers and 1 for inliers
    test_outliers = clf.predict(test_df)
    train_df['Outlier_prediction'] = train_outliers
    test_df['Outlier_prediction'] = test_outliers
    train_clean = train_df.drop(train_df[train_df['Outlier_prediction'] == -1].index, axis=0) 
    train_clean = train_clean.drop('Outlier_prediction', axis=1)  
    test_clean = test_df.drop(test_df[test_df['Outlier_prediction'] == -1].index, axis=0) 
    test_clean = test_clean.drop('Outlier_prediction', axis=1)  
    n_outliers_train = len(train_df) - len(train_clean)
    n_outliers_test = len(test_df) - len(test_clean)
    print('the number of outliers for the train set is:', n_outliers_train) 
    print('the number of outliers for the test set is:', n_outliers_test)
    return train_clean, test_clean

def standard_scaling(train_df, test_df):
    scaler = StandardScaler().fit(train_df)
    train_scaled = scaler.transform(train_df)
    test_scaled = scaler.transform(test_df)
    #Create dataframes with the correct index
    train_scaled = pd.DataFrame(train_scaled, index=train_df.index, columns=train_df.columns)
    test_scaled = pd.DataFrame(test_scaled, index=test_df.index, columns=test_df.columns)
    return train_scaled, test_scaled

def feature_reduction(train_df, test_df, var_threshold=0.05, corr_threshold=0.8):
    print('Number of features originally: ', len(train_df.columns))
    print("Removing Constant and Nearly Constant Features")
    var_threshold = VarianceThreshold(threshold=var_threshold)
    var_threshold.fit(train_df)
    indices_features_retained = var_threshold.get_support(indices=True)
    train_set_reduced1 = train_df.iloc[:, indices_features_retained]
    print('Features retained after elimination of constant and nearly constant features:', len(train_set_reduced1.columns))
    print("Removing Correlated Features")
    # Calculate the correlation matrix
    corr_matrix = train_set_reduced1.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []
    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            # If correlation exceeds the threshold
            if val >= corr_threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])
    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    train_set_reduced2 = train_set_reduced1.drop(columns=drops)
    print('Removed Columns {}'.format(drops))
    print('Number of features retained: ', len(train_set_reduced2.columns))
    test_set_reduced = test_df[train_set_reduced2.columns]
    return train_set_reduced2, test_set_reduced


def find_lowest_e_conf(smiles, num_conf=50, pruneRmsThresh=0.5):
    """
    Find the lowest energy conformer for a molecule with RDKit.
    
    :param smiles: str, SMILES string
    :param num_conf: int, number of conformers to search
    :return: str or None, XYZ coordinates of the lowest energy conformer or None if failed
    """
    results={}
    try:
        # Convert SMILES to RDKit molecule
        rdkmol = Chem.MolFromSmiles(smiles)
        if rdkmol is None:
            print(f"Skipping invalid SMILES: {smiles}")
            return None
        rdkmol = Chem.AddHs(rdkmol)
        rdForceFieldHelpers.MMFFSanitizeMolecule(rdkmol)
        # Set up embedding parameters
        params = AllChem.ETKDGv3()  # Use the ETKDG method
        params.pruneRmsThresh = 0.5  # Set RMSD pruning threshold
        params.randomSeed = 42  # Fix seed for reproducibility
        params.useRandomCoords = True  # Helps with embedding failures
        params.numThreads = -1 

        # Generate conformers
        conf_ids = AllChem.EmbedMultipleConfs(rdkmol, numConfs=num_conf, params=params) #Changed ETKGD --> ETKGDv3
        if len(conf_ids) == 0:
            print(f"Skipping {smiles} - No conformers generated.")
            return None
        # Optimize conformers
        results_MMFF = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(rdkmol, maxIters=1000, numThreads=-1)
        for i, result in enumerate(results_MMFF):
            results[i] = result[1]
        if not results:
            print(f"Skipping {smiles} - MMFF optimization failed for all conformers.")
            return None
        # Extract lowest energy conformer
        best_idx = min(results, key=results.get)
        structure = Chem.rdmolfiles.MolToXYZBlock(rdkmol, confId=best_idx)
        return structure
    except Exception as e:
        print(f"Skipping {smiles} due to error: {e}")
        return None

def read_db(path_to_database=None, name_target='y', smiles_col='smiles', debugging=False):
    print("Reading the database...")
    entire_db = pd.read_csv(path_to_database, index_col=0)
    print('The number of rows are:', len(entire_db))
    entire_db = entire_db[[smiles_col, name_target]].dropna(subset=[name_target])
    print('After dropping all rows without the target value:', len(entire_db))
    # #FOR DEBBUGGING ONLY
    if debugging:
        entire_db = entire_db.head(100)
    return entire_db