# !pip install yfinance
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

stock_symbols = ['GOOGL', 'AAPL', 'MSFT', 'AMZN', 'META']
# Download data for each stock and concatenate their data
stocks_data = {}
for symbol in stock_symbols:
    stocks_data[symbol] = yf.download(symbol, start='2010-01-01', end='2022-12-01')
data = pd.concat(stocks_data.values(), keys=stock_symbols)
data.drop('Volume', inplace=True, axis=1) #Volume not important for price prediction
lookback = 60 #lookback value for the LSTM


'''Preprocess the data
   Normalize the data and add time-sequential data to the x and y arrays
'''
def preProcess(data):
    # Normalize dataset from a range of 0 to 1
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for symbol in stock_symbols:
        stocks_df = data_scaled.loc[symbol]
        len_train_data = math.ceil(len(stocks_df) * 0.8)
        train_data = stocks_df.iloc[:len_train_data]
        test_data = stocks_df.iloc[len_train_data - lookback:]

        x_train_stock = []
        y_train_stock = []
        x_test_stock = []
        y_test_stock = []
        # print(test_data.columns)

        for i in range(lookback, len(train_data)):
            x_train_stock.append(train_data.iloc[i - lookback:i].values)
            y_train_stock.append(train_data['Close'].iloc[i])
        for j in range(lookback, len(test_data)):
            x_test_stock.append(test_data.iloc[j - lookback:j].values)
            y_test_stock.append(test_data['Close'].iloc[j])

        x_train.extend(x_train_stock)
        y_train.extend(y_train_stock)
        x_test.extend(x_test_stock)
        y_test.extend(y_test_stock)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    return x_train,y_train,x_test,y_test, data_scaled

'''Train an LSTM model
   Takes in the past 60 days (lookback) data and tries to predict the Close value for the 61th day'''
def trainModel(x_train,y_train):
    n_features = x_train.shape[2] #5 features [Open, High, Low, Close, Adj Close]
    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape=(lookback, n_features)))
    model.add(layers.LSTM(100, return_sequences=False))
    model.add(layers.Dense(25,activation='relu'))
    model.add(layers.Dense(1,activation='softmax'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(x_train, y_train, epochs=20, batch_size=32) #Train the model
    return model

def trainModelGRUNN(x_train,y_train):
    n_features = x_train.shape[2]
    model = keras.Sequential()
    model.add(layers.GRU(120, return_sequences=True, input_shape=(lookback, n_features)))
    model.add(layers.GRU(60, return_sequences=False))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(1, activation='softmax'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(x_train, y_train, epochs = 15, batch_size = 32)
    return model

def evalModel(x_test,y_test,model,data_scaled):
    # Create a separate scaler for the 'Close' column
    close_scaler = MinMaxScaler()
    close_scaler.fit(data.loc[:, ['Close']])
    # Make predictions on the test dataset
    y_pred = model.predict(x_test)

    # Denormalize the predicted values and actual values to get the original scale
    y_pred_orig = close_scaler.inverse_transform(y_pred)
    y_test_orig = close_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate the RMSE
    rmse = np.sqrt(np.mean((y_pred_orig - y_test_orig) ** 2))
    print("Root Mean Squared Error:", rmse)
    # Initialize the starting index for each stock's test data
    start_index = 0

    for symbol in stock_symbols:
        stocks_df = data_scaled.loc[symbol]
        len_train_data = math.ceil(len(stocks_df) * 0.8)
        test_data = stocks_df.iloc[len_train_data - lookback:]
        test_dates = data.loc[symbol].iloc[len_train_data:].index

        # Calculate the number of test samples for the current stock
        num_test_samples = len(test_data) - lookback

        # Extract the predicted and actual prices for the current stock
        stock_y_pred = y_pred_orig[start_index:start_index + num_test_samples].flatten()
        stock_y_test = y_test_orig[start_index:start_index + num_test_samples].flatten()

        # Plot the actual and predicted prices
        plt.figure(figsize=(15, 8))
        plt.plot(test_dates, stock_y_test, label=f'Actual {symbol} Stock Price')
        plt.plot(test_dates, stock_y_pred, label=f'Predicted {symbol} Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Closing Stock Price')
        plt.title(f'{symbol} Stock Price Prediction')
        plt.legend()
        plt.grid()
        plt.show()

        # Update the starting index for the next stock
        start_index += num_test_samples

x_train,y_train,x_test,y_test,data_scaled = preProcess(data)
model = trainModel(x_train,y_train)
# model = trainModelGRUNN(x_train,y_train)
evalModel(x_test, y_test, model, data_scaled)

# evalModel(x_test,y_test,model,data_scaled)

