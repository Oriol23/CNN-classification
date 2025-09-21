import math
import pandas as pd
from typing import Dict
from scipy.stats import pearsonr
from pandas.api.types import is_object_dtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance




def round_nnz(x,n_extra=0):
    """
    Round a number to the nearest non-zero decimal.

    Or the nearest pluts n_extra

    """
    if x == 0:
        return 0

    magnitude = math.floor(math.log10(abs(x))) # 10 ^ magnitude of the number

    # If number >= 1, round to nearest integer
    if magnitude >= 0:
        return round(x)

    # If number between 1 and 1e-4 round to nearest non-zero decimal
    decimals = abs(magnitude)
    return round(x, decimals+n_extra)


def calculate_importance(grouped_res,exp_mod=None) -> Dict: #type:ignore
    """Calculates feature importance and correlation.
    
    Takes the values of the hyperparameters used to train a model and the 
    training loss and calculates every hyperparameter's importance using 
    permutation importance with a random forest regresson, and the correlation 
    to the train loss using Pearson's coefficient. If exp_mod is specified, it
    calculates that for a single pair, otherwise it does so for all pairs.
    
    Args: 
        grouped_res: a dataframe with data that has been grouped based on two 
            columns, experiment and model.
        exp_mod: a tuple or list with a particular pair of experiment and model, 
            belonging to a valid pair of indices. 
        
    Returns:
        A dictionary containing the correlation and importance, separated by 
        experiment and model.
    """
    print("[INFO] Calculating importance...")
    importance_database = {}
    
    if exp_mod == None:
        pairs = grouped_res.indices.keys()
    else: 
        pairs = [exp_mod]

    for exp,mod in pairs:
        df = grouped_res.get_group((exp,mod))
        col_y = "train_loss"
        col_X = df.columns.to_list()[0:df.columns.to_list().index("train_loss")] 
        col_X.remove("Iter")
        col_X.append("Epoch #")
        # Hyperparameters assessed are all HP that change in value, except for Iter, 
        # and adding Epoch # 
        col_X[:] = [col for col in col_X if len(df[col].unique())>1]
        if col_X == []:
            importance_database.update({exp:{mod:{"Hyperparameters":[],
                                                "Correlation":[],
                                                "p-value":[],
                                                "Importance":[],
                                                "Uncertainty":[] }}})
            continue

        # if any of the hyperparameters is categorical we drop it since both tests 
        # are for numeric variables
        for hp in col_X:
            if is_object_dtype(df.loc[:,hp]):    
                col_X.remove(hp)

        X = df[col_X]       
        y = df[[col_y]]     
        r,p = pearsonr(X,y)
        randomForest = RandomForestRegressor(n_estimators=100,
                                            bootstrap=True,
                                            criterion='squared_error',
                                            random_state=0)
        randomForest.fit(X, y[col_y])
        #pochoimportance = randomForest.feature_importances_
        importance = permutation_importance(randomForest,
                                            X,y,
                                            random_state=0,n_repeats=20)
        importance_database.update(
            {exp:{mod:{"Hyperparameters":col_X,
                        "Correlation":r,
                        "p-value":p,
                        "Importance":importance["importances_mean"]/sum(importance["importances_mean"]),#type:ignore
                        "Uncertainty":importance["importances_std"]}}})
    return importance_database