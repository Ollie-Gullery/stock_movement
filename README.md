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
* Accuracy Score and Classfication Report graph outputs

# Example Output
Accuracy Score:


<img width="463" alt="Screen Shot 2023-01-25 at 9 57 38 PM" src="https://user-images.githubusercontent.com/115392875/214768733-7a37c704-a21b-4de6-9fd5-24924f97c9a0.png">




Confuson Matrix:


<img width="495" alt="Screen Shot 2023-01-25 at 9 55 59 PM" src="https://user-images.githubusercontent.com/115392875/214768652-aac13ab5-16e2-46b2-be86-717124b9837f.png">




Feature Improvement Plot:


<img width="495" alt="Screen Shot 2023-01-25 at 9 56 20 PM" src="https://user-images.githubusercontent.com/115392875/214768767-4d89b153-b603-4a3f-9f27-fde44ea7666b.png">



ROC Curve:


<img width="495" alt="Screen Shot 2023-01-25 at 9 56 26 PM" src="https://user-images.githubusercontent.com/115392875/214768770-8e0f888a-6201-4fce-a6e6-aec9685553e5.png">



Libraries Used Includes: `numpy`, `pandas`, `sklearn`, `matplotlib` 

# Potential Enhancements + future opportunties
* More user friendly interface
* Removing features with low impact on model to improve runtime
* Incorporate External Data
* Could further develop to provide stock recommendations to buy or sell certain stocks


