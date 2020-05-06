# -*- coding: utf-8 -*-
"""BTP-II FD004

# BTP Phase II - Dynamic Predictive Maintenance
# Authors - Vrishank Bhardwaj and Nikhil Panse

"""



import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score


def preprocess(df_rul, df_train, df_test):
    print("Preprocessing Data")

    #print(df_rul.head())
    #print(df_train.head())
    #print(df_test.head())

    ranges = [(0, 2), (6, 9), (11, 14), (15, 20), (21, 22), (24,26)]
    #print([item for start, end in ranges for item in df_train.columns[start:end]])
    df_train = df_train[[item for start, end in ranges for item in df_train.columns[start:end]]]

    ranges = [(0, 2), (6, 9), (11, 14), (15, 20), (21, 22), (24,26)]
    #print([item for start, end in ranges for item in df_test.columns[start:end]])
    df_test = df_test[[item for start, end in ranges for item in df_test.columns[start:end]]]

    print("Removing redundant columns due to extra spacing")
    df_train = df_train[df_train.columns[:26]]
    df_test = df_test[df_test.columns[:26]]

    # see data structure
    print('RUL: ', df_rul.shape)
    print('Train: ', df_train.shape)
    print('Test: ', df_test.shape)

    print("Preprocess Train")

    df_train.rename(columns = {0 : 'unit', 1 : 'cycle'}, inplace = True)

    total_cycles = df_train.groupby(['unit']).agg({'cycle' : 'max'}).reset_index()
    total_cycles.rename(columns = {'cycle' : 'total_cycles'}, inplace = True)
    print(df_train.head())
    df_train = df_train.merge(total_cycles, how = 'left', left_on = 'unit', right_on = 'unit')
    df_train['RUL'] = df_train.apply(lambda r: int(min(r['total_cycles'] - r['cycle'], 130)), axis = 1)


    df_train2 = df_train.copy()
    del df_train2['cycle']

    X_train = df_train2[df_train2.columns[:15]]
    print(X_train.head())
    y_train = df_train['RUL']

    print("Preprocess Test")

    # number of engines
    engines = df_test[0].unique()
    print("Number of engines = {}".format(len(engines)))


    df_list = []
    # get last cycle for each engine
    for i in engines:
        df = df_test[df_test[0]==i]
        last = (df[-1:])
        df_list.append(last)

    # union all rows in a dataframe
    X_test = pd.concat(df_list)
    del X_test[1]

    #print(df_test.columns)

    y_test = df_rul.values.flatten()

    y_train


    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    #X_test = sc.fit_transform(X_test)

    return X_train, X_test, y_train, y_test




def ContinousEval(X_train, y_train, name, splits=8):
    print("Continous Evaluation using {} batches".format(splits))
    X = X_train
    y = y_train

    tscv = TimeSeriesSplit(n_splits=splits)
    #print(tscv)

    train_rmse_scores=[]
    test_rmse_scores=[]

    train_r2_scores=[]
    test_r2_scores=[]

    for train_index, test_index in tscv.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print("Shape of training data : ", end="")
        print(X_train.shape)

        print("Shape of testing data : ", end="")
        print(X_test.shape)

        model = RandomForestRegressor(n_estimators=1050, max_depth=9, n_jobs=-1)

        print("Training model on batch")

        model.fit(X_train, y_train)

        print("Model succesfully trained")

        y_predicted_train = model.predict(X_train)
        y_predicted_test = model.predict(X_test)

        MSE_train = mean_squared_error(y_train, y_predicted_train)
        MSE_test = mean_squared_error(y_test, y_predicted_test)

        print('Train RMSE:', np.sqrt(MSE_train))
        train_rmse_scores.append(np.sqrt(MSE_train))
        print('Test RMSE:', np.sqrt(MSE_test))
        test_rmse_scores.append(np.sqrt(MSE_test))

        r2_train = r2_score(y_train, y_predicted_train)
        r2_test = r2_score(y_test, y_predicted_test)

        print('Train r2score:', r2_train)
        train_r2_scores.append(r2_train)
        print('Test r2score:', r2_test)
        test_r2_scores.append(r2_test)


    plt.subplot(2, 1, 1)
    plt.title('Continous Learning Evaluation for {}'.format(name))
    plt.plot(train_rmse_scores, label='Train RMSE')
    plt.plot(test_rmse_scores, label='Test RMSE')
    plt.ylabel('RMSE')
    #plt.xlabel('Batches Encountered')

    plt.subplot(2, 1, 2)
    plt.plot(train_r2_scores, label='Train r2 Score')
    plt.plot(test_r2_scores, label='Test r2 Score')
    plt.ylabel('r2 Score')
    plt.xlabel('Batches Encountered')
    plt.show()

    return 0




if __name__ == "__main__":

    for i in range(1,5):
        print("Continous Learning Evaluation for FD00{}".format(i))

        print("Loading FD00{}".format(i))
        df_rul = pd.read_table('CMAPSSData/RUL_FD00{}.txt'.format(i), header=None)
        df_train = pd.read_table('CMAPSSData/train_FD00{}.txt'.format(i), sep=' ', header=None)
        df_test = pd.read_table('CMAPSSData/test_FD00{}.txt'.format(i), sep=' ', header=None)

        X_train, X_test, y_train, y_test = preprocess(df_rul, df_train, df_test)
        ContinousEval(X_train, y_train, splits=8, name="FD00{}".format(i))