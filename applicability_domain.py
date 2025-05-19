from sklearn.model_selection import KFold
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, balanced_accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import make_scorer
from sklearn.svm import OneClassSVM
import pandas as pd
import matplotlib.pyplot as plt
from numpy import arange,array,ones
from scipy import stats
from CIMtools.applicability_domain import Box, Leverage, SimilarityDistance, TwoClassClassifiers, GPR_AD
import joblib  # For loading models

plt.rc('font', family='sans-serif')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (7, 6),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

def picture(name, y_test, y_pred, rmse_train, title, ad=None, desc_name = ''):
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, y_test)
    line = slope*np.array(y_pred)+intercept
    
    rmse = rmse_train
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")

    TI = 0 #True inliers (TP)
    FI = 0 #False inliers (FP)
    TO = 0 #True outliers (TN)
    FO = 0 #False outliers (FN)
    
    if name == 'without ad':
        ax.scatter(y_test, y_pred, color="blue", marker='.', s=150)
        OD = "NA"
    else:
        # The molecules for which the absolute prediction error is higher than 3×RMSE are identified as Y-outliers, 
        # while the rest are considered as Y-inliers. This will be considered as true AD
        y_in_ad = abs(y_test-y_pred) <= 3*rmse
        
        for num, (ad_pred, ad_true) in enumerate(zip(ad, y_in_ad)):
            # We consider AD definition approaches as binary classifiers returning True 
            # for X-inliers (within AD) and False for X-outliers (outside AD). 
            if ad_pred == ad_true:
                if ad_pred == 0: # TN
                    ax.scatter(y_test[num], y_pred[num], color="red", marker='s', s=50)
                    TO += 1
                else: # TP
                    ax.scatter(y_test[num], y_pred[num], color="blue", marker='.', s=25, alpha=0.5)
                    TI += 1
            else:
                if ad_pred == 0: # FN
                    ax.scatter(y_test[num], y_pred[num], color="red", marker='^', s=50)
                    FO += 1
                else: # FP
                    ax.scatter(y_test[num], y_pred[num], color="blue", marker='+', s=50, alpha=0.5)
                    FI += 1
        labels = ['TN', 'TP', 'FN', 'FP']
        colors = ['red', 'blue', 'red', 'blue']
        markers = ['s', '.', '^', '+']
        for i in range(4):
            ax.scatter([], [], label=labels[i], color=colors[i], marker=markers[i], s=150)
        ax.legend()
        OD = (TO/(TO+FI) + TI/(TI+FO))/2

    ax.set_xlabel("Experimental solv_oxidation_potential", color='black')
    ax.set_ylabel("Predicted solv_oxidation_potential", color='black')
    ax.set_title('{}'.format(title))

    plt.plot(line, y_pred, color='black', alpha=0.5)
    plt.plot(line-3*rmse, y_pred, 'k--', alpha=0.3)
    plt.plot(line+3*rmse, y_pred, 'k--',  alpha=0.3)

    plt.grid(True)
    plt.rc('xtick') 
    plt.rc('ytick')
    plt.savefig(f"AD/{desc_name}_{title}.png")

    return OD

# Create folders if they don't exist
os.makedirs("AD", exist_ok=True)

# List of dataset paths
paths_to_datasets = {
    'predictions/split10_desc_2D_best_predictions.csv': 'split10_desc_2D_SVR_best.pkl',
    'predictions/split10_desc_MFP_best_predictions.csv': 'split10_desc_MFP_SVR_best.pkl',
    'predictions/split10_desc_2D_3D_best_predictions.csv': 'split10_desc_2D_3D_SVR_best.pkl',

}

for path in paths_to_datasets.keys():
    dataset_name = os.path.basename(path).replace(".csv", "")  # Extract dataset name
    df = pd.read_csv(path, index_col=0)
    print(f"AD for {dataset_name} model: SVR")
    model = paths_to_datasets.get(path)
    model_filename = f"models/{model}"
    reg = joblib.load(model_filename)  # Load trained model
    print(f"Uploaded trained model: {model}")

    # Split into train and test
    train = df[df['split'] == 'train']
    test = df[df['split'] == 'test']

    y_train = train['solv_oxidation_potential']
    y_pred_train = train['y_pred_SVR']
    rmse_train = mean_squared_error(y_train, y_pred_train)

    y_test = test['solv_oxidation_potential']
    y_pred_test = test['y_pred_SVR']
    print('Q_squared and RMSE:', r2_score(y_test, y_pred_test), mean_squared_error(y_test, y_pred_test))

    X_train = train.iloc[:,:-7]
    X_test = test.iloc[:,:-7]

    picture(name='without ad', y_test=y_test, y_pred=y_pred_test, rmse_train=rmse_train, 
                title='Without Applicability Domain of Model', desc_name=dataset_name)

    #Method 1: Bounding Box
    print('Method 1: Bounding Box')
    AD_BB = Box().fit(X_train).predict(X_test)
    od = picture(name='with ad', y_test=y_test, y_pred=y_pred_test, rmse_train=rmse_train, title='With Bounding Box', ad=AD_BB,
                     desc_name=dataset_name)
    print('Outliers detection:', od)
    X_test_AD = X_test[AD_BB]
    y_test_AD = y_test[AD_BB]
    new_y_pred = reg.predict(X_test_AD)
    print('Q_squared and RMSE after BB:', r2_score(y_test_AD, new_y_pred), mean_squared_error(y_test_AD, new_y_pred))
    print('Fraction of molecules classified as inside AD:', (len(X_test_AD)/len(X_test)))

    #Method 2: Leverage
    print('Method 2: Leverage')
    leverage_model = Leverage(threshold='auto').fit(X_train, y_train)
    AD_Leverage = leverage_model.predict(X=X_test)
    od = picture(name='with ad', y_test=y_test, y_pred=y_pred_test, rmse_train=rmse_train, title='With Leverage', ad=AD_Leverage,
                     desc_name=dataset_name)
    print('Outliers detection:', od)
    X_test_AD = X_test[AD_Leverage]
    y_test_AD = y_test[AD_Leverage]
    new_y_pred = reg.predict(X_test_AD)
    print('Q_squared and RMSE after Leverage:', r2_score(y_test_AD, new_y_pred), mean_squared_error(y_test_AD, new_y_pred))
    print('Fraction of molecules classified as inside AD:', (len(X_test_AD)/len(X_test)))
            #find optimal threshold h
    AD_Lev_cv = Leverage(threshold='cv', score='ba_ad', 
                     reg_model=reg).fit(X_train, y_train).predict(X_test)
    od = picture(name='with ad', y_test=y_test, y_pred=y_pred_test, rmse_train=rmse_train, title='With Modified Leverage', ad=AD_Lev_cv,
                     desc_name=dataset_name)
    print('Outliers detection:', od)
    X_test_AD = X_test[AD_Lev_cv]
    y_test_AD = y_test[AD_Lev_cv]
    new_y_pred = reg.predict(X_test_AD)
    print('Q_squared and RMSE after CV Leverage:', r2_score(y_test_AD, new_y_pred), mean_squared_error(y_test_AD, new_y_pred))
    print('Fraction of molecules classified as inside AD:', (len(X_test_AD)/len(X_test)))

    #Method 3: Z1NN
    print('Method 3: Threshold distance')
    Z1NN_model = SimilarityDistance(threshold='auto').fit(X_train, y_train)
    AD_Z1NN = Z1NN_model.predict(X_test)
    od = picture(name='with ad', y_test=y_test, y_pred=y_pred_test, rmse_train=rmse_train, title='With Z1NN', ad=AD_Z1NN,
                     desc_name=dataset_name)
    print('Outliers detection:', od)
    X_test_AD = X_test[AD_Z1NN]
    y_test_AD = y_test[AD_Z1NN]
    new_y_pred = reg.predict(X_test_AD)
    print('Q_squared and RMSE after Threshold Distance:', r2_score(y_test_AD, new_y_pred), mean_squared_error(y_test_AD, new_y_pred))
    print('Fraction of molecules classified as inside AD:', (len(X_test_AD)/len(X_test)))
            #find optimal threshold with internal cv
    AD_Z1NN_cv = SimilarityDistance(score='ba_ad', threshold='cv', 
                                reg_model=reg).fit(X_train, y_train).predict(X_test)
    od = picture(name='with ad', y_test=y_test, y_pred=y_pred_test, rmse_train=rmse_train, title='With Modified Z1NN', ad=AD_Z1NN_cv,
                     desc_name=dataset_name)
    print('Outliers detection:', od)
    X_test_AD = X_test[AD_Z1NN_cv]
    y_test_AD = y_test[AD_Z1NN_cv]
    new_y_pred = reg.predict(X_test_AD)
    print('Q_squared and RMSE after CV Threshold Distance:', r2_score(y_test_AD, new_y_pred), mean_squared_error(y_test_AD, new_y_pred))
    print('Fraction of molecules classified as inside AD:', (len(X_test_AD)/len(X_test)))

    #Method 4: TwoClassClassifier
    print('Method 4: 2 Class Classifier')
    AD_2CC = TwoClassClassifiers(threshold='cv', score='ba_ad', reg_model=reg, 
                             clf_model=RandomForestClassifier(n_estimators=250, random_state=1, n_jobs=-1)).fit(X_train, y_train).predict(X_test)
    od = picture(name='with ad', y_test=y_test, y_pred=y_pred_test, rmse_train=rmse_train, title='With Two Class X inlier Y outlier Classifier', 
    ad=AD_2CC, desc_name=dataset_name)
    print('Outliers detection:', od)
    X_test_AD = X_test[AD_2CC]
    y_test_AD = y_test[AD_2CC]
    new_y_pred = reg.predict(X_test_AD)
    print('Q_squared and RMSE after CV TwoClassClassifier:', r2_score(y_test_AD, new_y_pred), mean_squared_error(y_test_AD, new_y_pred))
    print('Fraction of molecules classified as inside AD:', (len(X_test_AD)/len(X_test)))

    #Method 5: GPR
    print('Method 5: GPR')
    AD_GPR = GPR_AD(threshold='cv', score='ba_ad').fit(X_train, y_train).predict(X_test)
    od = picture(name='with ad', y_test=y_test, y_pred=y_pred_test, rmse_train=rmse_train, title='With GPR', ad=AD_GPR,
                     desc_name=dataset_name)
    print('Outliers detection:', od)
    X_test_AD = X_test[AD_GPR]
    y_test_AD = y_test[AD_GPR]
    new_y_pred = reg.predict(X_test_AD)
    print('Q_squared and RMSE after CV GPR:', r2_score(y_test_AD, new_y_pred), mean_squared_error(y_test_AD, new_y_pred))
    print('Fraction of molecules classified as inside AD:', (len(X_test_AD)/len(X_test)))

    final_df = pd.DataFrame({'Box': AD_BB, 'Leverage': AD_Leverage, 'CV Leverage': AD_Lev_cv,
                                 'Z1NN': AD_Z1NN, 'CV Z1NN': AD_Z1NN_cv, '2CC': AD_2CC,
                                 'GPR': AD_GPR}, index=test.index)
    final_df.to_csv(f'AD/AD_{dataset_name}.csv')


