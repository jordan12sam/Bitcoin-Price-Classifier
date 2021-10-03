import pandas as pd


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