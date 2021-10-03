import itertools

import pandas as pd
import numpy as np
from talib import abstract, get_function_groups
from sklearn.ensemble import RandomForestClassifier


FEE = 0

#open csv as dataframe
def get_data(filename="historical_data.csv"):
    historical_data = pd.read_csv(filename)
    historical_data.set_index("closetime", inplace=True)
    return historical_data

#add a target column to the the table containing the direction of the future price movement
def classify_data(df):

    #given the current price and the future price, return the direction of the price movement
    def classify_record(current, future):
        if float(future) * (1-FEE)**2 > float(current):
            return 1
        elif float(future) * (1-FEE)**2 < float(current):
            return 0

    
    #future price column holds the price in the next row down from the current price
    df = df
    df["future"] = df["close"].shift(-1)
    df["target"] = list(map(classify_record, df["close"], df["future"]))
    df.drop(["future"], axis=1, inplace=True)
    targets = df["target"]
    df.drop(["target"], axis=1, inplace=True)
    df.dropna(inplace=True)
    targets.dropna(inplace=True)
    return targets

#return a dataframe of indicators given a dataframe of ochlv data
#by default gives a large amount of indicators
#specific indicators can be selected
def get_indicators(df, indicators_list=None):

    if not indicators_list:
        functions_dict = get_function_groups()
        functions_dict = { key: functions_dict[key] for key in ["Momentum Indicators",
                                                                "Overlap Studies", 
                                                                "Price Transform", 
                                                                "Volume Indicators",
                                                                "Statistic Functions"] }

        indicators_list = list(itertools.chain.from_iterable(functions_dict.values()))
        for indicator in ["MAVP"]:
            indicators_list.remove(indicator)
    
    indicators_df = pd.DataFrame()
    for indicator in indicators_list:
        func = abstract.Function(indicator)
        func_data = func(df)
        if isinstance(func_data, pd.DataFrame):
            for column in func_data.columns:
                indicators_df[f"{indicator}-{column}"] = func_data[column]
        else:
            indicators_df[indicator] = func_data
    indicators_df.dropna(axis=0, inplace=True)
    return indicators_df

#calculate the log returns of the data to make it stationary
def log_returns(df):
    for column in df.columns:
        df[column] = np.log(df[column]) - np.log(df[column].shift(1))
    df.dropna(axis=0, inplace=True)
    return df

#standardise indicators to mean 0, std 1
def standardise(df):
    for column in df.columns:
        mean = df[column].mean()
        std = df[column].std()
        df[column] = [(x-mean)/std for x in df[column]]
    return df

#rank most important features in predicting the future price using a random forest classifier
def feature_importance(X,y,X_columns):
    model = RandomForestClassifier()
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feats = []
    importance = []
    for f in range(X.shape[1]):
        if f > 0:
            feats.append(X_columns[indices[f]])
            importance.append(importances[indices[f]])
    mask = [x > 0.015 for x in importance]
    return [feats[i] for i in range(len(feats)) if mask[i]]

#keep only independent features
#removes any features that are too strongly correlated
def drop_dependent_features(df):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    df = df.drop(to_drop, axis=1)
    return df.columns