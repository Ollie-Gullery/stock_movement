import os
import time

import numpy as np
import pandas as pd
import yfinance as yf
import xlsxwriter

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import plot_tree

import seaborn as sns



ticker_list = ['JPM', 'AAPL', 'AMZN', 'WMT']  # large array of businesses



# Data Preprocessing

def grab_price_data(ticker_list):
    price_df = pd.DataFrame()
    df_list = []
    for ticker in ticker_list:
        data = yf.download(ticker, start='2021-01-01', end='2022-12-17')
        data['ticker'] = ticker
        df_list.append(pd.DataFrame(data))
    for df in df_list:
        price_df = pd.concat([price_df, df], ignore_index=False)
    price_df.to_csv(f"price_data.csv")


if os.path.exists('price_data.csv'):

    price_data = pd.read_csv("price_data.csv", index_col=False)
else:
    grab_price_data(ticker_list)

    price_data = pd.read_csv('price_data.csv', index_col=False)

# data reorganisation
price_data = price_data[['ticker', 'Date', 'Close', 'Adj Close', 'High', 'Low', 'Open', 'Volume']]

# Sort data on symbol then datetime
price_data.sort_values(by=['ticker', 'Date'], inplace=True, ignore_index=True)

price_data['change_in_adj_close'] = price_data['Adj Close'].diff()

ticker_change = price_data["ticker"] != price_data['ticker'].shift(1)

price_data['change_in_adj_close'] = np.where(ticker_change == True, np.nan, price_data['change_in_adj_close'])

# print(price_data[price_data.isna().any(axis=1)]) -> Checks to ensure that change_in_adj_close = null

# End of Data Preprocessing

# Start of Developing Momentum Indicators

# Momentum Indicators (5 total)
# 1. Stochastic Oscillator

n = 14

low_14, high_14 = price_data[['ticker', 'Low']].copy(), price_data[['ticker', 'High']].copy()

low_14 = low_14.groupby('ticker')['Low'].transform(lambda x: x.rolling(window=n).min())
high_14 = high_14.groupby('ticker')['High'].transform(lambda x: x.rolling(window=n).max())

k_percent = 100 * ((price_data['Adj Close'] - low_14) / (high_14 - low_14))
price_data['low_14'] = low_14
price_data['high_14'] = high_14
price_data['k_percent'] = k_percent

# 2. Relative Strength Indicator

# 14-day time frame
n = 14

up_df, down_df = price_data[['ticker', 'change_in_adj_close']].copy(), price_data[
    ['ticker', 'change_in_adj_close']].copy()

up_df.loc['change_in_adj_close'] = up_df.loc[(up_df['change_in_adj_close'] < 0), 'change_in_adj_close'] = 0

down_df.loc['change_in_adj_close'] = down_df.loc[(up_df['change_in_adj_close'] > 0), 'change_in_adj_close'] = 0

down_df['change_in_adj_close'] = down_df['change_in_adj_close'].abs()

# Calculate Exponential Weighted Moving Average (ewma) to apply weight to data that is more current
ewma_up = up_df.groupby('ticker')['change_in_adj_close'].transform(lambda x: x.ewm(span=n).mean())
ewma_down = down_df.groupby('ticker')['change_in_adj_close'].transform(lambda x: x.ewm(span=n).mean())

# Relative Strength
relative_strength = ewma_up / ewma_down

# Relative Strength Index
relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

price_data['RSI'] = relative_strength_index

# 3. Williams Percentage Range

r_percent = ((high_14 - price_data['Adj Close']) / (high_14 - low_14)) * -100

price_data['r_percent'] = r_percent

# Moving Average Convergence Divergence (MACD)

# signal line sell or buy signal

ema_26 = price_data.groupby('ticker')['Adj Close'].transform(lambda x: x.ewm(span=26).mean())
ema_12 = price_data.groupby('ticker')['Adj Close'].transform(lambda x: x.ewm(span=12).mean())
macd = ema_12 - ema_26

ema_9_macd = macd.ewm(span=9).mean()

price_data['MACD'] = macd
price_data['MACD_EMA'] = ema_9_macd

# 4. Price Rate of Change
n = 9

price_data['Price_Rate_of_Change'] = price_data.groupby('ticker')['Adj Close'].transform(
    lambda x: x.pct_change(periods=n))


# 5. On Balance Volume

def obv(group):
    # Grab volume and adj close volume
    volume = group['Volume']
    change = group['Adj Close'].diff()
    prev_obv = 0
    obv_values = []

    for i, j in zip(change, volume):
        if i > 0:
            current_obv = prev_obv + j
        elif i < 0:
            current_obv = prev_obv - j
        else:
            current_obv = prev_obv
        # OBV.append(current_obv)
        prev_obv = current_obv
        obv_values.append(current_obv)
    return pd.Series(obv_values, index=group.index)


obv_groups = price_data.groupby('ticker').apply(obv)

price_data['On Balance Volume'] = obv_groups.reset_index(level=0, drop=True)

# End of Momentum Indicators

# data cleaning

closed_groups = price_data.groupby('ticker')['Adj Close']

closed_groups = closed_groups.transform(lambda x: x.shift(1) < x)

price_data['Prediction'] = closed_groups * 1

price_data = price_data.dropna()

# Building the Model

X_Cols = price_data[['RSI', "k_percent", 'r_percent', 'Price_Rate_of_Change', 'On Balance Volume']]
Y_Cols = price_data['Prediction']

x_train, x_test, y_train, y_test = train_test_split(X_Cols, Y_Cols, random_state=0)

# Random Forest Classifer
rand_frst_clf = RandomForestClassifier(n_estimators=100, oob_score=True, criterion='gini', random_state=0)

# Fit Data
rand_frst_clf.fit(x_train, y_train)

"""
Plotting the whole forest
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
axes = axes.ravel()
for i, estimator in enumerate(rand_frst_clf.estimators_[:9]):
    ax = axes[i]
    ax.set_title(f'Tree {i}')
    plot_tree(estimator, filled=True, ax = ax)
plt.savefig("random_forest.png)
"""

# Plotting Individual Tree

tree = rand_frst_clf.estimators_[0]


# plt.figure(figsize=(10,10))
# plot_tree(tree, filled=True)
# plt.savefig("tree.svg")

def save_tree_to_desktop(your_mac_username, tree):
    if os.path.exists(f"/Users/{your_mac_username}/Desktop/stock_decision_tree.png") or your_mac_username =="pass":
        pass
    else:
        fig = Figure(figsize=(10, 10))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot()
        plot_tree(tree, filled=True, ax=ax)
        fig.savefig(f"/Users/{your_mac_username}/Desktop/stock_decision_tree.png", dpi=600)  # to save do your desktop


your_mac_username = input("Type 'pass' to skip or To save image of tree from random forest (for mac) input your username e.g JohnSmith: ")

save_tree_to_desktop(your_mac_username, tree)

# if os.path.exists('tree.png'):
#     pass
# else:
#     tree = rand_frst_clf.estimators_[0]
#     plt.figure(figsize=(10, 10))
#     plot_tree(tree, filled=True)
#     plt.savefig("tree.png")

# Make Predictions
y_pred = rand_frst_clf.predict(x_test)
acc_score = accuracy_score(y_test, rand_frst_clf.predict(x_test), normalize=True) * 100.0
# Data Validation/Evaluation
print(f'Correct Prediction (%): {acc_score}')

target_names = ['Down Day', 'Up Day']

report = classification_report(y_true=y_test, y_pred=y_pred, target_names=target_names, output_dict=True)

report_df = pd.DataFrame(report).transpose()

print(report_df)

# Model Evaluation: Confusion Matrix

rf_matrix = confusion_matrix(y_test, y_pred)

# Normalise
rf_matrix_normalised = rf_matrix.astype(float) / rf_matrix.sum(axis=1)[:, np.newaxis]
fix, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(rf_matrix_normalised, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Normalised Confusion Matrix')
plt.show()

# Feature Engineering/Evaluation

feature_imp = pd.Series(rand_frst_clf.feature_importances_, index=X_Cols.columns).sort_values(ascending=False)

print(f'\nThis explains how much each indicator contributes to the model:\n{feature_imp} \nIf we used '
      f'larger data sets it could be worth removing features which are not as meaningful/useful to the model')

x_values = list(range(len(rand_frst_clf.feature_importances_)))
cumulative_importances = np.cumsum(feature_imp.values)

plt.plot(x_values, cumulative_importances, 'g-')

# At 95% of Importance Retained
plt.hlines(y=0.95, xmin=0, xmax=len(feature_imp), color='red', linestyles='dashed')

# Format x ticks
plt.xticks(x_values, feature_imp.index, rotation=30, fontsize=5)

# labels
plt.xlabel('Variable')
plt.ylabel('Cumulative Importance')
plt.title('Random Forest Feature Importance at 95% Importance Retained')
plt.show()

# Receiver Operating Characteristic Curve (ROC Curve)
# You could use this curve to adjust False positives or negatives according to your risk profile and strategy
y_pred_proba = rand_frst_clf.predict_proba(x_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Roc curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC Curve)')
plt.legend(loc="lower right")
plt.show()


plt.plot(x_test, y_test)
plt.plot(x_test, y_pred)
plt.title('Predicted vs Actual Movement')
plt.xlabel('Price')
plt.ylabel('Time')
plt.show()

def model_improvement(x_train, y_train, x_test, y_test, acc_score):
    # Number of trees in a random forest
    n_estimators = list(range(200, 2000, 200))

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', None, 'log2']

    # Maximum number of levels in tree
    max_depth = list(range(10, 110, 10))

    min_samples_split = [2, 5, 10, 20, 30, 40]

    min_samples_leaf = [1, 2, 7, 12, 14, 16, 20]

    bootstrap = [True, False]

    # Random Grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap
                   }

    print(f'\n{random_grid}')

    rf = RandomForestClassifier()

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5,
                                   verbose=5, random_state=42, n_jobs=-1)

    start_time = time.time()
    rf_random.fit(x_train, y_train)
    end_time = time.time()
    print(f"Time Taken: {end_time - start_time}")

    y_pred_2 = rf_random.predict(x_test)
    improved_acc_score = accuracy_score(y_test, y_pred_2, normalize=True) * 100.0
    print(f'Improved* Correct Prediction (%): {improved_acc_score}')

    target_names = ['Down Day', 'Up Day']

    report = classification_report(y_true=y_test, y_pred=y_pred_2, target_names=target_names, output_dict=True)

    report_df = pd.DataFrame(report).transpose()

    print(report_df)

    # Model Evaluation: Confusion Matrix

    rf_matrix = confusion_matrix(y_test, y_pred_2)

    # Normalise
    rf_matrix_normalised = rf_matrix.astype(float) / rf_matrix.sum(axis=1)[:, np.newaxis]
    fix, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(rf_matrix_normalised, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Improved* Normalised Confusion Matrix')
    plt.show()

    # Feature Importance Analysis
    feature_imp = pd.Series(rand_frst_clf.feature_importances_, index=X_Cols.columns).sort_values(ascending=False)

    print(f'\nThis explains how much each indicator contributes to the model:\n{feature_imp} \nIf we used '
          f'larger data sets it could be worth removing features which are not as meaningful/useful to the model')

    x_values = list(range(len(rand_frst_clf.feature_importances_)))
    cumulative_importances = np.cumsum(feature_imp.values)

    plt.plot(x_values, cumulative_importances, 'g-')

    # At 95% of Importance Retained
    plt.hlines(y=0.95, xmin=0, xmax=len(feature_imp), color='red', linestyles='dashed')

    # Format x ticks
    plt.xticks(x_values, feature_imp.index, rotation=30, fontsize=5)

    # labels
    plt.xlabel('Variable')
    plt.ylabel('Cumulative Importance')
    plt.title('Improved* Random Forest Feature Importance at 95% Importance Retained')
    plt.show()

    # Roc Curve
    y_pred_proba = rand_frst_clf.predict_proba(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Roc curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC Curve)')
    plt.legend(loc="lower right")
    plt.show()

    print(f'Accuracy Increased By: {improved_acc_score - acc_score}% - from {acc_score}% to {improved_acc_score}%.')


# model_improvement(x_train, y_train, x_test, y_test, acc_score)
