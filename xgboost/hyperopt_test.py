

# Familiar imports
import numpy as np
import pandas as pd

# For ordinal encoding categorical variables, splitting data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import warnings
warnings.filterwarnings('ignore')

from xgboost.callback import EarlyStopping

#Import 'scope' from hyperopt in order to obtain int values for certain hyperparameters.
from hyperopt.pyll.base import scope



X_train = np.loadtxt("X_train.txt")
X_test = np.loadtxt("X_test.txt")
y_train = np.loadtxt("y_train.txt")[:,1]
y_test = np.loadtxt("y_test.txt")[:,1]



#Split the data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=3, train_size=.8)


#Define the space over which hyperopt will search for optimal hyperparameters.
space = {'max_depth': scope.int(hp.quniform("max_depth", 1, 5, 1)),
        'gamma': hp.uniform ('gamma', 0,1),
        'reg_alpha' : hp.uniform('reg_alpha', 0,50),
        'reg_lambda' : hp.uniform('reg_lambda', 10,100),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0,1),
        'min_child_weight' : hp.uniform('min_child_weight', 0, 5),
        'n_estimators': 10000,
        'learning_rate': hp.uniform('learning_rate', 0, .15),
        'tree_method':'gpu_hist', 
        'gpu_id': 0,
        'random_state': 5,
        'max_bin' : scope.int(hp.quniform('max_bin', 200, 550, 1))}



##Define the hyperopt objective.
#def hyperparameter_tuning(space):
#    model = xgb.XGBRegressor(eval_metrix = "rmse", **space)
#    
#    #Define evaluation datasets.
#    evaluation = [(X_train, y_train), (X_valid, y_valid)]
#    
#    #Fit the model. Define evaluation sets, early_stopping_rounds, and eval_metric.
#    model.fit(X_train, y_train,
#              eval_set=evaluation, 
#              early_stopping_rounds=100,  # Early stopping rounds here
#              verbose=False) 
#    
#    #Obtain prediction and rmse score.
#    pred = model.predict(X_valid)
#    rmse = mean_squared_error(y_valid, pred, squared=False)
#    print ("SCORE:", rmse)
#    
#    #Specify what the loss is for each model.
#    return {'loss':rmse, 'status': STATUS_OK, 'model': model}

def hyperparameter_tuning(space):
    # Create a DMatrix, which is the recommended format for XGBoost training
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    
    # Define evaluation datasets
    evaluation = [(dtrain, 'train'), (dvalid, 'valid')]

    # Update the space to match the parameters required by xgb.train()
    params = {
        'eval_metric': 'rmse',
        'tree_method': 'hist',  # Using 'hist' for CPU
        'random_state': 5,
        **space
    }
    
    # Train using xgb.train, which supports early stopping
    model = xgb.train(params, dtrain, 
                      num_boost_round=10000,  # Maximum number of boosting iterations
                      evals=evaluation,
                      early_stopping_rounds=100,  # Early stopping based on RMSE
                      verbose_eval=False)  # Control verbosity

    # Make predictions on the validation set
    pred = model.predict(dvalid)
    rmse = mean_squared_error(y_valid, pred, squared=False)
    print("SCORE:", rmse)
    
    return {'loss': rmse, 'status': STATUS_OK, 'model': model}





#Run 20 trials.
trials = Trials()
best = fmin(fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=30,
            trials=trials)

print(best)




#Create instace of best model.
best_model = trials.results[np.argmin([r['loss'] for r in 
    trials.results])]['model']

#Examine model hyperparameters
print(best_model)




xgb_preds_best = best_model.predict(X_valid)
xgb_score_best = mean_squared_error(y_valid, xgb_preds_best, squared=False)
print('RMSE_Best_Model:', xgb_score_best)

xgb_standard = xgb.XGBRegressor().fit(X_train, y_train)
standard_score = mean_squared_error(y_valid, xgb_standard.predict(X_valid), squared=False)
print('RMSE_Standard_Model:', standard_score)

