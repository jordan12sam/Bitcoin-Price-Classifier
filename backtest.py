import pipeline

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, cohen_kappa_score


#simulate a trading strategy given bitcoin prices
#return the value of the investment over the given period
#simple simulation whereby you are either fully invested or not at all
def backtest(strategy, prices):
    fee = pipeline.FEE
    #value of investments held in btc/usd
    btc = 0
    usd = 100
    strategy_index = []

    for minute, price in enumerate(prices):
        btc = btc * (1 + price)
        strategy_index.append(btc + usd)

        if strategy[minute] == 0:
            usd = btc*(1-fee) + usd
            btc = 0
        elif strategy[minute] == 1:
            btc = btc + usd*(1-fee)
            usd = 0


    return strategy_index

if __name__ == "__main__":
    print("loading model...")

    file = '2021-10-03--17-43-12'
    model = tf.keras.models.load_model(f'models/{file}.hd5')

    print("loading data...")

    train_x, train_y, train_ts, val_x, val_y, val_ts = pipeline.create_model_data()
    prices = pipeline.price_history(val_ts)

    predicted = np.argmax(model.predict(val_x), axis=1)
    hold = np.ones((len(predicted)))
    random = np.random.randint(2, size=(len(predicted)))

    print("running backtest...")

    #run four different strategies:
    #'hold' is only buys, ie it just follows the price of bitcoin
    #'model' is the machine learning model
    #'optimal' uses the target data. ie always buys before an increase, sells before a decrease
    #'random' is just a random agent. buys/sells randomly
    hold_index = backtest(hold, prices)
    model_index = backtest(predicted, prices)
    optimal_index = backtest(val_y, prices)
    random_index = backtest(random, prices)

    print("Strategy Results:")
    print(f"f1-score: {f1_score(val_y, predicted)}")
    print(f"Hold: {hold_index[-1]}")
    print(f"Model: {model_index[-1]}")
    print(f"Optimal: {optimal_index[-1]}")
    print(f"Random: {random_index[-1]}")

    plt.plot(model_index, c='blue')
    plt.plot(hold_index, c='red')
    #plt.plot(optimal_index, c='black')
    #plt.plot(random_index, c='green')
    plt.ylabel('USD')
    plt.xlabel('Days')
    #plt.yscale(value='log')
    plt.show()