# stock_movement
A small project I did that scrapes https://finance.yahoo.com/ based on a list of input tickers (default is 'JPM', 'AAPL', 'AMZN', 'WMT') and uses Random Forest to classify upwards or downwards movements.

## Overview
Historical data from selected stocks was stored into a csv file `price_data.csv`. Five momentum indicators (stochastic oscillator, RSI, Williams percentage range, 
price rate of change and on the balance volume) utlised to estimate movement of stocks. 

A classification report was the completed using a Accuracy Score (number of accurate predictions the model made on the test set),
Normalised Confusion Matrix, Receiver Operating Characteristic Curve, and feature importance graph. Improvements where then made through using `RandomisedSearchCV` in which we use a wide range of possible values for each hyperparameter and then using cross-validation (at k = 5) to try estimate the optimal hyperparameters.

## Implementation

First open terminal and type `cd` into the directory where you would like to download this program. Then, clone the stock_movement repository.

```
git clone https://github.com/Ollie-Gullery/stock_movement.git
```
Use the following command to run the program:
```
python movement_predict.py
```

# Output
* csv file of the scraped data (`price_data.csv`)
* Accuracy Score and graph outputs

# Example Output
Accuracy Score:
<img width="979" alt="" src="">

Confuson Matrix:
<img width="979" alt="" src="">

Feature Improvement Plot:

ROC Curve:
