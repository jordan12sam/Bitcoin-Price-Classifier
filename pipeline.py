import itertools
from collections import deque
from random import random, shuffle
from datetime import datetime

import pandas as pd
import numpy as np
from talib import abstract, get_function_groups
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


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

#sort data into sequences
def sequence_data(data, targets):
    sequential_data = []
    #sequence length
    window = 15
    #deque acts like a moving window
    prev_days = deque(maxlen=window)

    for timestamp in data.index:
        prev_days.append([data[column][timestamp] for column in data.columns])
        if len(prev_days) == window:
            sequential_data.append([np.array(prev_days), 
                                    targets[timestamp], 
                                    timestamp])
    return np.array(sequential_data)

#split the data into training and validation sets
#splits in a 1:0.2 ratio
def validation_split(sequences):
    proportion = 0.2
    split = int(sequences.shape[0]*proportion)
    train = sequences[:-split]
    val = sequences[-split:]
    return train, val

#balance the amount of buys and sells
def balance_data(sequential_data):
    buys = []
    sells = []

    for seq, target, timestamp in sequential_data:
        if target == 0:
            sells.append(np.array([seq, target, timestamp]))
        elif target == 1:
            buys.append(np.array([seq, target, timestamp]))

    lower = min(len(buys), len(sells))
    buys, sells = buys[:lower], sells[:lower]
    balanced_data = buys + sells
    shuffle(balanced_data)

    sequential_data = []
    for seq, target, timestamp in balanced_data:
        sequential_data.append([seq, target, timestamp])

    return np.array(sequential_data)

#split sequences from their target
def target_split(target_data):
    x_seq, y_seq, ts_seq = target_data[:,0], target_data[:,1], target_data[:,2]

    #get a list of sequences
    x = []
    for i in range(x_seq.shape[0]):
        sequence = []
        for j in range(x_seq[i].shape[0]):
            record = []
            for k in range(x_seq[i][j].shape[0]):
                record.append(x_seq[i][j][k])
            sequence.append(record)
        x.append(sequence)

    #get a list of buys/sells
    y = []
    for i in range(y_seq.shape[0]):
        y.append(y_seq[i])

    #get a list of timestamps
    ts = []
    for i in range(ts_seq.shape[0]):
        ts.append(ts_seq[i])

    return x, y, ts

#select best indicators to train with
def select_feats_from_historical_data():
    data = get_data() 
    targets = classify_data(data)
    indicators = get_indicators(data)

    indicators = standardise(indicators)
    data = log_returns(data)

    timestamps = set(data.index).intersection(set(indicators.index), set(targets.index))
    data = data.loc[timestamps]
    indicators = indicators.loc[timestamps]
    targets = targets[timestamps]

    feats = feature_importance(indicators.values, targets, indicators.columns)
    feats = drop_dependent_features(indicators.loc[:, feats])
    feats_list = set([feat.split('-')[0] for feat in feats])
    return feats_list

#full pipeline from raw data to ready for training
def create_model_data():
    indicators_list = select_feats_from_historical_data()
    
    data = get_data()
    targets = classify_data(data)
    indicators = get_indicators(data, indicators_list)

    indicators = standardise(indicators)
    data = log_returns(data)

    timestamps = data.index & indicators.index & targets.index
    data = data.loc[timestamps]
    indicators = indicators.loc[timestamps]
    targets = targets[timestamps]

    data = data.join(indicators, how="inner")
    data = sequence_data(data, targets)
    
    train, val = validation_split(data)
    train = balance_data(train)

    train_x, train_y, train_ts = target_split(train)
    val_x, val_y, val_ts = target_split(val)

    return train_x, train_y, train_ts, val_x, val_y, val_ts

#train the model
def train_model():

    #get data to train model
    train_x, train_y, train_ts, val_x, val_y, val_ts = create_model_data()

    #define model structure
    #an lstm is used
    data_shape = (len(train_x[0]), len(train_x[0][0]))
    model = Sequential([
        LSTM(units=128, input_shape=data_shape, activation="relu", return_sequences=True),
        Flatten(),
        Dense(units=2, activation="softmax")
    ])

    opt = Adam(lr=0.02, decay=1e-6)

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    tensorboard = TensorBoard(log_dir=f"logs/{time}")
    checkpoint = ModelCheckpoint("models/{}.hd5".format(f"{time}", monitor='val_accuracy', verbose=0, save_best_only=True, mode='max'))

    history = model.fit(
        train_x, train_y,
        batch_size = 1,
        epochs = 16,
        validation_data=(val_x, val_y),
        callbacks = [tensorboard, checkpoint],
        shuffle = False)

#get the price history as a list of percentage change in price
def price_history(mask=True):
    data = get_data()
    data = data["close"].pct_change()
    data.dropna(axis=0, inplace=True)
    return data[mask]