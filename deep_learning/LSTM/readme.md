## Stock Prediction Project
This project covers the basics of LSTM modelling which here is used for predicting the closing price of a stock by
looking at the previous data gathered for the stock.

The LSTM model looks at the closing prices of "N" previous days to predict the closing price of next day and so on.

1. `stock_prediction.py` - The file user can execute to feed in the dataset, train the model and get the required output
2. `Figure_1.png` - The plot which can be used to compare the actual closing price and the predicted closing price on
the validation data
3. `training.csv` - The dataset used for training. The columns are in following order: 
    
    1. Date 
    2. Day open
    3. Day high
    4. Day low
    5. Day close
    6. Day last
    7. Total trade quantity