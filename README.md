# Advanced Stock Price Prediction Model Development

## Project Overview
This project involves the development of an advanced stock price prediction model using TensorFlow and Keras. The model leverages Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) neural networks to predict the stock prices of five major tech companies: Google, Apple, Microsoft, Amazon, and Meta. The model is trained on historical stock data spanning from 01/01/2010 to 12/01/2022.

## Data Preprocessing
The data, which includes Open, High, Low, Close, and Adjusted Close prices, is normalized and sequentially stored for each stock. Each data point contains information from the past 60 days.

## Model Architecture
The LSTM model is constructed with two LSTM layers (300 and 150 units respectively), followed by two Dense layers (80 and 25 units respectively), and a final Dense layer with linear activation for output. The GRU-RNN model is similarly structured, but with GRU layers instead of LSTM. Both models are trained for 20 epochs with a batch size of 32, using Adam as the optimizer and Mean Squared Error as the loss function.

## Model Evaluation
The models are evaluated based on their Root Mean Squared Error (RMSE) on the test data. The LSTM model achieved an RMSE of 6.611062022357579, while the GRU-RNN model achieved a lower RMSE of 5.262483405205591, indicating higher accuracy in predicting stock prices.

## Visualization
The project also includes a visualization component, where the actual and predicted stock prices are plotted against time, providing a clear visual representation of the model's performance.

## Libraries Used
The project code is written in Python, utilizing libraries such as yfinance for data acquisition, pandas and numpy for data manipulation, sklearn for data preprocessing, and matplotlib for data visualization.

## Conclusion
This project demonstrates the potential of neural networks in financial forecasting and highlights the importance of data preprocessing and model tuning in achieving accurate predictions. It also showcases proficiency in handling time-series data, training and tuning neural network models, and interpreting model performance metrics.

## Usage
To use this project, clone the repository and install the necessary dependencies. Run the Python script to train the model and visualize the results. Adjust the parameters as needed to optimize the model's performance.

## Future Work
Future work could involve tuning the model parameters further to improve accuracy, incorporating additional features into the model, or applying the model to other types of time-series prediction tasks.
