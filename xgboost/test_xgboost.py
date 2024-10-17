import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

#def load_data(features_folder, targets_folder):
#    feature_files = sorted(os.listdir(features_folder))
#    target_files = sorted(os.listdir(targets_folder))
#    
#    # Initialize lists to store features and targets
#    features_list = []
#    targets_list = []
#
#    for feature_file, target_file in zip(feature_files, target_files):
#        # Load feature data
#        feature_path = os.path.join(features_folder, feature_file)
#        with open(feature_path, 'r') as f:
#            features = np.array([float(x) for x in f.read().split()])
#            features_list.append(features)
#        
#        # Load target data
#        target_path = os.path.join(targets_folder, target_file)
#        with open(target_path, 'r') as f:
#            targets = np.array([float(x) for x in f.read().split()])
#            targets_list.append(targets)
#    
#    # Convert to numpy arrays
#    X = np.nan_to_num(np.array(features_list), nan = 0.0, posinf = 0.0, neginf = 0.0)
#    y = np.array(targets_list)
#    
#    return X, y
#
#
## Load the training data
#features_folder = '../../dmg-train/features'
#targets_folder = '../../dmg-train/measures'
#X_train, y_train = load_data(features_folder, targets_folder)
#
#
## Load the testing data
#features_folder = '../../dmg-test/features'
#targets_folder = '../../dmg-test/measures'
#X_test, y_test = load_data(features_folder, targets_folder)
#
#np.savetxt("X_test.txt",X_test)
#np.savetxt("y_test.txt",y_test)
#np.savetxt("X_train.txt",X_train)
#np.savetxt("y_train.txt",y_train)

# Function to calculate mean relative error
def mean_relative_error(y_true, y_pred):
    # Avoid division by zero by using a small epsilon where y_true is zero
    epsilon = 1e-10
    relative_errors = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), epsilon, None))
    return np.mean(relative_errors)

X_train = np.loadtxt("X_train.txt")
X_test = np.loadtxt("X_test.txt")
y_train = np.loadtxt("y_train.txt")
y_test = np.loadtxt("y_test.txt")

#models = []
#for i in range(y_train.shape[1]):
#    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, max_depth=6, learning_rate=0.1)
#    model.fit(X_train, y_train[:, i])
#    models.append(model)
#
## Predict and evaluate for each target
#for i in range(y_test.shape[1]):
#    y_pred = models[i].predict(X_test)
#    mse = mean_squared_error(y_test[:, i], y_pred)
#    print(f'Target {i+1}: MSE = {mse:.4f}')


# Define the parameter grid
param_grid = {
    'n_estimators': [1000,10000,25000],
    'max_depth': [3,4],
    'learning_rate': [0.0005,0.001,0.0001],
    'subsample': [0.8],
    'colsample_bytree': [0.9],
    'gamma': [0],
    'reg_alpha': [1],
    'reg_lambda': [0]
}
# Perform grid search for each target
best_models = []

for i in range(y_train.shape[1]):
    model = xgb.XGBRegressor(objective='reg:squarederror')
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train[:, i])
    best_models.append(grid_search.best_estimator_)


        # Print the best parameters for this target
    print(f"Best parameters for target {i+1}: {grid_search.best_params_}")
    
    ## Get cross-validation results
    #cv_results = grid_search.cv_results_
    #
    ## Print MSE (mean_test_score is negative MSE, so multiply by -1)
    #for mean_score, std_score, params in zip(cv_results['mean_test_score'], 
    #                                         cv_results['std_test_score'], 
    #                                         cv_results['params']):
    #    mse = -mean_score  # Convert negative MSE back to positive MSE
    #    print(f"Params: {params}, MSE = {mse:.4f} (std = {std_score:.4f})")

    model = best_models[i]
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test[:, i], y_pred)
    mre = mean_relative_error(y_test[:, i], y_pred)
    print(f"Target {i+1}: MSE on test set= {mse:.4f}")
    print(f"Target {i+1}: MRE on test set = {mre:.5f}")





## Evaluate the best models
#for i, model in enumerate(best_models):
#    y_pred = model.predict(X_test)
#    mse = mean_squared_error(y_test[:, i], y_pred)
#    mre = mean_relative_error(y_test[:, i], y_pred)
#    print(f"Target {i+1}: MSE on test set= {mse:.4f}")
#    print(f"Target {i+1}: MRE on test set = {mre:.5f}")
#
#    break
