import pandas as pd
import time

#get candlestick data for a symbol over a given time period
def get_klines(symbol, start, end):
    historical_data = pd.DataFrame()
    while end > start:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=1000&startTime={int(start)}&endTime={int(min(end, start + 8.64e10))}"
        print(url)
        data = pd.read_json(url)
        data.columns = ['opentime', 'open', 'high', 'low', 'close', 'volume', 
                        'closetime', 'quote asset volume', 'number of trades',
                        'taker by base', 'taker buy quote', 'ignore']
        data = data[['closetime', 'open', 'high', 'low', 'close', 'volume']]
        historical_data = pd.concat([historical_data, data], axis=0, ignore_index=True, keys=None)
        start += 8.64e10
    historical_data.set_index('closetime', inplace=True)
    historical_data = historical_data[~historical_data.index.duplicated(keep='first')]
    return historical_data

#save data to csv
if __name__ == "__main__":
    data = get_klines("BTCUSDT", 1451606400000, time.time() * 1000)
    data.to_csv("historical_data.csv", index=True)
    print("done")