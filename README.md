# Bitcoin Price Classifier

This project aimed to create an LSTM neural network trained to predict movements in bitcoin prices using candlestick data and technical analysis indicators created from this data.

Data is downloaded to a csv in [download_data.py](https://github.com/jordan12sam/bitcoin_price_classifier/blob/master/download_data.py) using the binance API. 

All data preprocessing and model training is in [pipeline.py](https://github.com/jordan12sam/bitcoin_price_classifier/blob/master/pipeline.py).

And finally the model can be tested in [backtest.py](https://github.com/jordan12sam/bitcoin_price_classifier/blob/master/backtest.py).

An accuracy of 0.62 and f1 score of 0.6 was acheived using this method. In testing the model was generally able to match and often even slighlty outperform the market. See the graph below whereby the model outperforms the market by about 65 percentage point; giving a ROI of 144% over a period where bitcoin prices increased by 79%.

&nbsp;
![lines](images/lines.png)  

