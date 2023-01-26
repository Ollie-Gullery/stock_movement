# stock_movement
A python program that scrapes https://finance.yahoo.com/ based on a list of input tickers (default is 'JPM', 'AAPL', 'AMZN', 'WMT') and uses Random Forest to classify upwards or downwards movements.

#Overview
Historical data from selected stocks was stored into a csv file [price_data.csv]. Five momentum indicators (stochastic oscillator, RSI, Williams percentage range, 
price rate of change and on the balance volume) utlised to estimate movement of stocks. 

A classification report was the completed using a Accuracy Score (number of accurate predictions the model made on the test set),
Normalised Confusion Matrix, Receiver Operating Characteristic Curve, and feature importance graph. Improvements where then made through using 
[RandomisedSearchCV] in which we use a wide range of possible values for each hyperparameter and then using cross-validation (at k = 5) to try estimate the optimal hyperparameters.
