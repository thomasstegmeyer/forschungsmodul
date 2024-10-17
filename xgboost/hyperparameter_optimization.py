import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Load your training and test data
X_train = np.loadtxt("X_train.txt")
X_test = np.loadtxt("X_test.txt")
y_train = np.loadtxt("y_train.txt")
y_test = np.loadtxt("y_test.txt")

# Define the XGBoost regressor
xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', n_jobs=-1)

# Define the parameter search space for Bayesian Optimization
param_space = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'gamma': Real(0, 5),
    'reg_alpha': Real(0, 10),
    'reg_lambda': Real(0.1, 10),
}

# Initialize Bayesian optimization using BayesSearchCV
opt = BayesSearchCV(
    estimator=xgb_regressor,
    search_spaces=param_space,
    n_iter=50,  # Number of parameter settings that are sampled
    scoring='neg_mean_squared_error',  # Optimize for MSE
    cv=3,  # 3-fold cross-validation
    n_jobs=-1,
    verbose=0
)


for i in range(9):
    # Fit the model
    opt.fit(X_train, y_train[:,i])

    # Get the best model and print the best hyperparameters
    best_model = opt.best_estimator_

    # Predict on the test set
    y_pred = best_model.predict(X_test)

    # Calculate RMSE on the test set
    mse = mean_squared_error(y_test[:,i], y_pred)
    print(f"Target {i} Test MSE: {mse}")
    print("Best hyperparameters:", opt.best_params_)
