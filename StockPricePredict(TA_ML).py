import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
tf.keras.utils.set_random_seed(1)
import xgboost as xgb
import datetime
import warnings
warnings.filterwarnings("ignore")
from scipy.signal import argrelextrema
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from keras.layers import GRU, LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model

# Defining a Sidebar
st.sidebar.title("Stock Price Prediction :")
st.sidebar.write('Copyright by BB_Chan')
code = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL,0005.hk,...) :")
start = st.sidebar.date_input("Select Start Date",value=datetime.date(2019,1,1),
                      min_value=datetime.date(2000,1,1),
                      max_value=datetime.date(2024,7,1))
end = st.sidebar.date_input("Select End Date",value=datetime.date(2024,12,31))
signal_days = st.sidebar.number_input("Select Trading Signal Days :",1,10,5,1)
# Defining Checkbox
st.sidebar.write('Select Technical Indicator(s) :')
MACD_DMI = st.sidebar.checkbox('Trend : MACD & DMI')
RSI_KDJ = st.sidebar.checkbox('Momentun : RSI & KDJ')
BB_BIAS = st.sidebar.checkbox('Volatility : BB & BIAS')
# Defining Checkbox
st.sidebar.write('Select Prediction Model(s) :')
XGBoost = st.sidebar.checkbox('XGBoost')
New_GRU = st.sidebar.checkbox('Create new GRU')
Rel_GRU = st.sidebar.checkbox('Load saved GRU')
New_LSTM = st.sidebar.checkbox('Create new LSTM')
Rel_LSTM = st.sidebar.checkbox('Load saved LSTM')
New_LSTM_AM = st.sidebar.checkbox('Create new LSTM - Attention Mechanism')
Rel_LSTM_AM = st.sidebar.checkbox('Load saved LSTM - Attention Mechanism')
New_LSTM_FEAT = st.sidebar.checkbox('Create new LSTM - Features')
Rel_LSTM_FEAT = st.sidebar.checkbox('Load saved LSTM - Features')
# Defining a Button
button = st.sidebar.button('Submit')
if not button:
    st.stop()

stock = yf.download(code, start, end)
stock.to_csv(code + '.csv')
df = pd.read_csv(code + '.csv')
df['Close'] = round(df['Close'],2)
st.header(code)
st.subheader('Stock Data')
st.dataframe(df)

# Calculate Moving Averages
df['SMA10'] = df['Close'].rolling(window=10).mean()
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['EMA10'] = df['Close'].ewm(span=10).mean()
df['EMA50'] = df['Close'].ewm(span=50).mean()
df['EMA100'] = df['Close'].ewm(span=100).mean()

# Calculate Bollinger BANDS
def BBANDS(df0, ma_days):
    ma = df0.Close.rolling(window=ma_days).mean()
    sd = df0.Close.rolling(window=ma_days).std()
    df0['MiddleBand'] = ma
    df0['UpperBand'] = ma + (std_dev * sd)
    df0['LowerBand'] = ma - (std_dev * sd)
    return df0
ma_days = 20
std_dev = 2
df = BBANDS(df, ma_days)

# Calculate Moving Average Convergence Divergence
df['EMA1'] = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
df['EMA2'] = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
df['DIF'] = df['EMA1'] - df['EMA2']
df['DEA'] = df['DIF'].ewm(span=9, adjust=False, min_periods=9).mean()
df['MACD'] = 2 * (df['DIF'] - df['DEA'])

# Calculate Directional Movement Index
def get_adx(high, low, close, lookback):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(lookback).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / lookback).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1 / lookback).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha=1 / lookback).mean()
    return plus_di, minus_di, adx_smooth
df['Plus_di'] = pd.DataFrame(get_adx(df['High'], df['Low'], df['Close'], 14)[0]).rename(columns={0: 'Plus_di'})
df['Minus_di'] = pd.DataFrame(get_adx(df['High'], df['Low'], df['Close'], 14)[1]).rename(columns={0: 'Minus_di'})
df['ADX'] = pd.DataFrame(get_adx(df['High'], df['Low'], df['Close'], 14)[2]).rename(columns={0: 'ADX'})

# Calculate KDJ
def calKDJ(df2):
    df2['MinLow'] = df2['Low'].rolling(9, min_periods=9).min()
    df2['MinLow'].fillna(value=df2['Low'].expanding().min(), inplace=True)
    df2['MaxHigh'] = df2['High'].rolling(9, min_periods=9).max()
    df2['MaxHigh'].fillna(value=df2['High'].expanding().max(), inplace=True)
    df2['RSV'] = (df2['Close'] - df2['MinLow']) / (df2['MaxHigh'] - df2['MinLow']) * 100
    for i in range(len(df2)):
        if i == 0:
            df2.loc[i, 'K'] = 50
            df2.loc[i, 'D'] = 50
        if i > 0:
            df2.loc[i, 'K'] = df2.loc[i - 1, 'K'] * 2 / 3 + 1 / 3 * df2.loc[i, 'RSV']
            df2.loc[i, 'D'] = df2.loc[i - 1, 'D'] * 2 / 3 + 1 / 3 * df2.loc[i, 'K']
            df2.loc[i, 'J'] = 3 * df2.loc[i, 'K'] - 2 * df2.loc[i, 'D']
    return df2
df = calKDJ(df)

# Calculate Relative Strength Index
def calRSI(df3, periodList):
    df3['diff'] = df3['Close'] - df3['Close'].shift(1)
    df3['diff'].fillna(0, inplace=True)
    df3['up'] = df3['diff']
    df3['up'][df3['up'] < 0] = 0
    df3['down'] = df3['diff']
    df3['down'][df3['down'] > 0] = 0
    for period in periodList:
        df3['upAvg' + str(period)] = df3['up'].rolling(period).sum() / period
        df3['upAvg' + str(period)].fillna(0, inplace=True)
        df3['downAvg' + str(period)] = abs(df3['down'].rolling(period).sum() / period)
        df3['downAvg' + str(period)].fillna(0, inplace=True)
        df3['RSI' + str(period)] = 100 - 100 / (df3['upAvg' + str(period)] / df3['downAvg' + str(period)] + 1)
    return df3
periods = [6, 12, 24]
df = calRSI(df, periods)

# Calculate BIAS
def calBIAS(df1, periodList):
    # Review periods: 6, 12 & 24days
    for period in periodList:
        df1['MA'+str(period)] = df1['Close'].rolling(window=period).mean()
        df1['MA'+str(period)].fillna(value=df['Close'], inplace=True)
        df1['BIAS'+str(period)] = (df1['Close'] - df1['MA'+str(period)])/df1['MA'+str(period)]*100
    return df1
periods = [6, 12, 24]
df = calBIAS(df, periods)

# Calculate Support & Resistance
WINDOW = 30
df['min'] = round((df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=WINDOW)[0]]['Close']),2)
df['max'] = round((df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=WINDOW)[0]]['Close']),2)

# ###################################
# ###  Calculate & print EMA10 Buy Points:
cnt = len(df)-3
EMA10buyDate = ''
while cnt > len(df)-signal_days:
    # Rule 1：'Close' continuously increase 3 days
    if (df.iloc[cnt]['Close'] < df.iloc[cnt+1]['Close']) & (df.iloc[cnt+1]['Close'] < df.iloc[cnt+2]['Close']):
        # Rule 2：EMA10 continuously increase 3 days
        if (df.iloc[cnt]['EMA10'] < df.iloc[cnt+1]['EMA10']) & (df.iloc[cnt+1]['EMA10'] < df.iloc[cnt+2]['EMA10']):
            # Rule 3：'Close' of 3rd day higher than EMA10
            if (df.iloc[cnt+1]['EMA10'] > df.iloc[cnt]['Close']) & (df.iloc[cnt+2]['EMA10'] < df.iloc[cnt+1]['Close']):
                EMA10buyDate = EMA10buyDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt - 1
# ###  Calculate & print EMA10 Sell Points:
cnt = len(df)-3
EMA10sellDate = ''
while cnt > len(df)-signal_days:
    # Rule 1：'Close' continuously decrease 3 days
    if (df.iloc[cnt]['Close'] > df.iloc[cnt+1]['Close']) & (df.iloc[cnt+1]['Close'] > df.iloc[cnt+2]['Close']):
        # Rule 2：EMA10 continuously decrease 3 days
        if (df.iloc[cnt]['EMA10'] > df.iloc[cnt+1]['EMA10']) & (df.iloc[cnt+1]['EMA10'] > df.iloc[cnt+2]['EMA10']):
            # Rule 3：'Close' of 3rd day lower than EMA10
            if (df.iloc[cnt+1]['EMA10'] < df.iloc[cnt]['Close']) & (df.iloc[cnt+2]['EMA10'] > df.iloc[cnt+1]['Close']):
                EMA10sellDate = EMA10sellDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt - 1
# ###################################
# ###  Calculate & print EMA50 Buy Points:
cnt = len(df)-3
EMA50buyDate = ''
while cnt > len(df)-signal_days:
    # Rule 1：'Close' continuously increase 3 days
    if (df.iloc[cnt]['Close'] < df.iloc[cnt+1]['Close']) & (df.iloc[cnt+1]['Close'] < df.iloc[cnt+2]['Close']):
        # Rule 2：EMA50 continuously increase 3 days
        if (df.iloc[cnt]['EMA50'] < df.iloc[cnt+1]['EMA50']) & (df.iloc[cnt+1]['EMA50'] < df.iloc[cnt+2]['EMA50']):
            # Rule 3：'Close' of 3rd day lower than EMA50
            if (df.iloc[cnt+1]['EMA50'] > df.iloc[cnt]['Close']) & (df.iloc[cnt+2]['EMA50'] < df.iloc[cnt+1]['Close']):
                EMA50buyDate = EMA50buyDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt - 1
# ###  Calculate & print EMA50 Sell Points:
cnt = len(df)-3
EMA50sellDate = ''
while cnt > len(df)-signal_days:
    # Rule 1，'Close' continuously decrease 3 days
    if (df.iloc[cnt]['Close'] > df.iloc[cnt+1]['Close']) & (df.iloc[cnt+1]['Close'] > df.iloc[cnt+2]['Close']):
        # Rule 2，EMA50 continuously decrease 3 days
        if (df.iloc[cnt]['EMA50'] > df.iloc[cnt+1]['EMA50']) & (df.iloc[cnt+1]['EMA50'] > df.iloc[cnt+2]['EMA50']):
            # Rule 3，'Close' of 3rd day lower than EMA50
            if (df.iloc[cnt+1]['EMA50'] < df.iloc[cnt]['Close']) & (df.iloc[cnt+2]['EMA50'] > df.iloc[cnt+1]['Close']):
                EMA50sellDate = EMA50sellDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt - 1
# ###################################
# ###  Calculate & print EMA100 Buy Points:
cnt = len(df)-3
EMA100buyDate = ''
while cnt > len(df)-signal_days:
    # Rule 1：'Close' continuously increase 3 days
    if (df.iloc[cnt]['Close'] < df.iloc[cnt+1]['Close']) & (df.iloc[cnt+1]['Close'] < df.iloc[cnt+2]['Close']):
        # Rule 2：EMA100 continuously increase 3 days
        if (df.iloc[cnt]['EMA100'] < df.iloc[cnt+1]['EMA100']) & (df.iloc[cnt+1]['EMA100'] < df.iloc[cnt+2]['EMA100']):
            # Rule 3：'Close' of 3rd day lower than EMA100
            if (df.iloc[cnt+1]['EMA100'] > df.iloc[cnt]['Close']) & (df.iloc[cnt+2]['EMA100'] < df.iloc[cnt+1]['Close']):
                EMA100buyDate = EMA100buyDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt - 1
# ###  Calculate & print EMA100 Sell Points:
cnt = len(df)-3
EMA100sellDate = ''
while cnt > len(df)-signal_days:
    # Rule 1，'Close' continuously decrease 3 days
    if (df.iloc[cnt]['Close'] > df.iloc[cnt+1]['Close']) & (df.iloc[cnt+1]['Close'] > df.iloc[cnt+2]['Close']):
        # Rule 2，EMA100 continuously decrease 3 days
        if (df.iloc[cnt]['EMA100'] > df.iloc[cnt+1]['EMA100']) & (df.iloc[cnt+1]['EMA100'] > df.iloc[cnt+2]['EMA100']):
            # Rule 3，'Close' of 3rd day lower than EMA100
            if (df.iloc[cnt+1]['EMA100'] < df.iloc[cnt]['Close']) & (df.iloc[cnt+2]['EMA100'] > df.iloc[cnt+1]['Close']):
                EMA100sellDate = EMA100sellDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt - 1
# ###################################
# ###  Calculate & print BBANDS Buy Points:
cnt = len(df)-1
BBbuyDate = ''
while cnt > len(df)-signal_days:
    if cnt >= 30:
        if (df.iloc[cnt-1]['Close'] > df.iloc[cnt]['LowerBand']) & (df.iloc[cnt]['Close'] < df.iloc[cnt]['LowerBand']):
            BBbuyDate = BBbuyDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt-1
# ###  Calculate & print BBANDS Sell Points:
cnt = len(df)-1
BBsellDate = ''
while cnt > len(df)-signal_days:
    if cnt >= 30:
        if (df.iloc[cnt-1]['Close'] < df.iloc[cnt-1]['UpperBand']) & (df.iloc[cnt]['Close'] > df.iloc[cnt]['UpperBand']):
            BBsellDate = BBsellDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt-1
# ###################################
# ###  Calculate & print MACD Buy Points:
cnt = len(df)-1
MACDbuyDate = ''
while cnt > len(df)-signal_days:
    if cnt >= 30:
        # Rule 1：Current day DIF > DEA
        if (df.iloc[cnt]['DIF'] > df.iloc[cnt]['DEA']) & (df.iloc[cnt-1]['DIF'] < df.iloc[cnt-1]['DEA']):
            # Rule 2：Red bar appears, then MACD > 0
            if df.iloc[cnt]['MACD'] > 0:
                MACDbuyDate = MACDbuyDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt-1
# ###  Calculate & print MACD Sell Points:
cnt = len(df)-1
MACDsellDate = ''
while cnt > len(df)-signal_days:
    if cnt >= 30:
        # Rule 1：Current day DIF< DEA
        if (df.iloc[cnt]['DIF'] < df.iloc[cnt]['DEA']) & (df.iloc[cnt-1]['DIF'] > df.iloc[cnt-1]['DEA']):
            # Rule 2：Bar shows downward trend
            if df.iloc[cnt]['MACD'] < df.iloc[cnt-1]['MACD']:
                MACDsellDate = MACDsellDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt-1
# ###################################
# ###  Calculate & print DMI Buy Points:
cnt = len(df)-1
DMIbuyDate = ''
while cnt > len(df)-signal_days:
    if cnt >= 30:
        if (df.iloc[cnt-1]['ADX'] < 25) & (df.iloc[cnt]['ADX'] > 25) & (df.iloc[cnt]['Plus_di'] > df.iloc[cnt]['Minus_di']):
            DMIbuyDate = DMIbuyDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt-1
# ###  Calculate & print DMI Sell Points:
cnt = len(df)-1
DMIsellDate = ''
while cnt > len(df)-signal_days:
    if cnt >= 30:
        if (df.iloc[cnt-1]['ADX'] < 25) & (df.iloc[cnt]['ADX'] > 25) & (df.iloc[cnt]['Plus_di'] < df.iloc[cnt]['Minus_di']):
            DMIsellDate = DMIsellDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt-1
# ###################################
# ###  Calculate & print BIAS Buy Points:
cnt = len(df)-1
BIASbuyDate = ''
while cnt > len(df)-signal_days:
    if cnt >= 3:
        # Rule 1：
        if df.iloc[cnt]['BIAS12'] <= -7:
            BIASbuyDate = BIASbuyDate + df.iloc[cnt]['Date'] + ', '
            # Rule 2：
            if (df.iloc[cnt]['BIAS6'] > df.iloc[cnt]['BIAS24']) & (df.iloc[cnt-1]['BIAS6'] < df.iloc[cnt-1]['BIAS24']):
                BIASbuyDate = BIASbuyDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt - 1
# ###  Calculate & print BIAS Sell Points:
cnt = len(df)-1
BIASsellDate = ''
while cnt > len(df)-signal_days:
    if cnt >= 3:
        # Rule 1：
        if df.iloc[cnt]['BIAS12'] >= 7:
            BIASsellDate = BIASsellDate + df.iloc[cnt]['Date'] + ', '
            # Rule 2 :
            if (df.iloc[cnt]['BIAS6'] < df.iloc[cnt]['BIAS24']) & (df.iloc[cnt - 1]['BIAS6'] > df.iloc[cnt - 1][
                    'BIAS24']):
                    BIASsellDate = BIASsellDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt - 1
# ####################################
# ###  Calculate & print KDJ Buy Points:
cnt = len(df)-1
KDJbuyDate = ''
while cnt > len(df)-signal_days:
    if cnt >= 3:
        # Rule 1：Last day J > 10, cuurent day < 10
        if (df.iloc[cnt]['J'] < 10) & (df.iloc[cnt - 1]['J'] > 10):
            KDJbuyDate = KDJbuyDate + df.iloc[cnt]['Date'] + ', '
            cnt = cnt - 1
            continue
        # Rule 2：Both K & D < 20, K > D shows golden crossover
        # Rule 1 or Rule 2, if satisfied Rule 1, then directly continues
        if (df.iloc[cnt]['K'] > df.iloc[cnt]['D']) & (df.iloc[cnt - 1]['D'] > df.iloc[cnt - 1]['K']):
            # then both K & D < 20
            if (df.iloc[cnt]['K'] < 20) & (df.iloc[cnt]['D'] < 20):
                KDJbuyDate = KDJbuyDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt - 1
# ###  Calculate & print KDJ Sell Points:
cnt = len(df)-1
KDJsellDate = ''
while cnt > len(df)-signal_days:
    if cnt >= 3:
        # Rule 1：Last day J < 100, current day > 100
        if (df.iloc[cnt]['J'] > 100) & (df.iloc[cnt - 1]['J'] < 100):
            KDJsellDate = KDJsellDate + df.iloc[cnt]['Date'] + ', '
            cnt = cnt - 1
            continue
        # Rule 2：Both K & ,D > 80, K < D shows dealth crossover
        if (df.iloc[cnt]['K'] < df.iloc[cnt]['D']) & (df.iloc[cnt - 1]['D'] < df.iloc[cnt - 1]['K']):
            # then both K & D > 80
            if (df.iloc[cnt]['K'] > 80) & (df.iloc[cnt]['D'] > 80):
                KDJsellDate = KDJsellDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt - 1
# ###################################
# ###  Calculate & print RSI Buy Points:
cnt = len(df)-1
RSIbuyDate = ''
while cnt > len(df)-signal_days:
    if cnt >= 3:
        # Rule 1：Current Day RSI6 < 20
        if df.iloc[cnt]['RSI6'] < 20:
            # Rule 2.1：Current day RSI6 > RSI12
            if (df.iloc[cnt]['RSI6'] > df.iloc[cnt]['RSI12']) & (df.iloc[cnt - 1]['RSI6'] < df.iloc[cnt - 1]['RSI12']):
                RSIbuyDate = RSIbuyDate + df.iloc[cnt]['Date'] + ', '
                # Rule 2.2：Current day RSI6 > RSI24
                if (df.iloc[cnt]['RSI6'] > df.iloc[cnt]['RSI24']) & (df.iloc[cnt - 1]['RSI6'] < df.iloc[cnt - 1]['RSI24']):
                    RSIbuyDate = RSIbuyDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt - 1
# ###  Calculate & print RSI Sell Points:
cnt = len(df)-1
RSIsellDate = ''
while cnt > len(df)-signal_days:
    if cnt >= 3:
        # Rule 1：Current day RSI6 > 80
        if df.iloc[cnt]['RSI6'] < 80:
            # Rule 2.1：Current day RSI6 < RSI12
            if (df.iloc[cnt]['RSI6'] < df.iloc[cnt]['RSI12']) & (df.iloc[cnt - 1]['RSI6'] > df.iloc[cnt - 1]['RSI12']):
                RSIsellDate = RSIsellDate + df.iloc[cnt]['Date'] + ', '
                # Rule 2.2：Current day RSI6< RSI24
                if (df.iloc[cnt]['RSI6'] < df.iloc[cnt]['RSI24']) & (df.iloc[cnt - 1]['RSI6'] > df.iloc[cnt - 1]['RSI24']):
                    if RSIsellDate.index(df.iloc[cnt]['Date']) == -1:
                        RSIsellDate = RSIsellDate + df.iloc[cnt]['Date'] + ', '
    cnt = cnt - 1
# ###################################
# ###  Calculate & print Support Levels:
cnt = len(df)-1
SupportDate = ''
while cnt > len(df)-90:
    if cnt >= 30:
        if (df.iloc[cnt]['min'] > 0):
            SupportDate = SupportDate + str(df.iloc[cnt]['min']) + ' (' + df.iloc[cnt]['Date'] + '), '
    cnt = cnt-1

# ###  Calculate & print Resistance Levels:
cnt = len(df)-1
ResistDate = ''
while cnt > len(df)-60:
    if cnt >= 30:
        if (df.iloc[cnt]['max'] > 0):
            ResistDate = ResistDate + str(df.iloc[cnt]['max']) + ' (' + df.iloc[cnt]['Date'] + '), '
    cnt = cnt-1

# ### Develop X_ & y_train & test data
length_df = len(df)
split_ratio = 0.8  # %80 train + %20 test
length_train = round(length_df * split_ratio)
length_test = length_df - length_train
test_start = df.iloc[length_train].name
training_set = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(training_set)
# Separating the data
training_set = df['Close'].iloc[:test_start].values
test_set = df['Close'].iloc[test_start:].values
prediction_days = 50
X_train = []
y_train = []
for i in range(prediction_days, len(training_set_scaled)):
    X_train.append(training_set_scaled[i - prediction_days: i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# Pre-processing the data
dataset_total = pd.concat((df["Close"].iloc[:test_start], df["Close"].iloc[test_start:]), axis=0)
inputs = dataset_total[len(dataset_total) - len(test_set) - prediction_days:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
# Predict the values
X_test = []
for i in range(prediction_days,len(inputs)):
    X_test.append(inputs[i-prediction_days:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

# ### Evaluate & print metrics
def Calculate_print_metrics(test,predict):
    mape = mean_absolute_percentage_error(test, predict)
    rmse = mean_squared_error(test, predict, squared=False)
    mae = mean_absolute_error(test, predict)
    # Print the evaluation metrics and directional accuracy
    st.write('MAPE                   :', str(round(mape,6)))
    st.write('Root mean squared error:', str(round(rmse, 6)))
    st.write('Mean absolute error    :', str(round(mae, 6)))

# ###################################
# Print Current Date & Stock Price
st.write('Close Price of Current Date ' + df.iloc[(len(df)-1)]['Date'] + ' is ' + str(df.iloc[(len(df)-1)]['Close']))
Last_Close_Price = df.iloc[(len(df)-1)]['Close']
# Plot Charts
# EMA Chart
st.subheader('Close Prices w/ Exponential Moving Average (Trend)')
lines_chart1 = px.line(df, x="Date", y=["Close", "EMA10", "EMA50", "EMA100"],
                           color_discrete_map={'Close': 'goldenrod','EMA10': 'blue','EMA50': 'green',
                                               'EMA100': 'purple'})
st.plotly_chart(lines_chart1)
st.write('EMA10  Buy Signal  : ', EMA10buyDate)
st.write('EMA10  Sell Signal : ', EMA10sellDate)
st.write('EMA50  Buy Signal  : ', EMA50buyDate)
st.write('EMA50  Sell Signal : ', EMA50sellDate)
st.write('EMA100 Buy Signal  : ', EMA100buyDate)
st.write('EMA100 Sell Signal : ', EMA100sellDate)

# Candlestick Chart
st.subheader('Candlestick Chart')
candlestick = go.Candlestick(x=df['Date'],
                             open=df['Open'], high=df['High'], low=df['Low'],
                             close=df['Close'], name='Candlestick')
candlestick_layout = go.Layout()
candlestick_fig = go.Figure(data=candlestick, layout=candlestick_layout)
st.plotly_chart(candlestick_fig)

# Volume Chart
bar_graph0 = px.bar(df, x=df['Date'],y=df['Volume']/1000000)
st.subheader('Volume')
st.plotly_chart(bar_graph0)

# Display Technical Indicators
if MACD_DMI :
# MACD Chart
    bar_graph1 = px.bar(df, x=df['Date'],y=df['MACD'])
    st.subheader('Moving Average Convergence Divergence (Trend)')
    st.plotly_chart(bar_graph1)
    st.write('MACD Buy Signal  : ', MACDbuyDate)
    st.write('MACD Sell Signal : ', MACDsellDate)
# DMI Chart
    lines_chart4 = px.line(df, x=df['Date'],y=[df['Plus_di'],df['Minus_di'],df['ADX']])
    st.subheader('Directional Movement Index (Trend)')
    st.plotly_chart(lines_chart4)
    st.write('DMI  Buy Signal  : ', DMIbuyDate)
    st.write('DMI  Sell Signal : ', DMIsellDate)

if RSI_KDJ :
# RSI Chart
    lines_chart6 = px.line(df, x=df['Date'],y=[df['RSI6'],df['RSI12'],df['RSI24']])
    st.subheader('Relative Strength Index (Momentum)')
    st.plotly_chart(lines_chart6)
    st.write('RSI  Buy Signal  : ', RSIbuyDate)
    st.write('RSI  Sell Signal : ', RSIsellDate)
# KDJ Chart
    lines_chart5 = px.line(df, x=df['Date'],y=[df['K'],df['D']])
    st.subheader('KDJ (Momentum)')
    st.plotly_chart(lines_chart5)
    st.write('KDJ  Buy Signal  : ', KDJbuyDate)
    st.write('KDJ  Sell Signal : ', KDJsellDate)

if BB_BIAS :
# BB Chart
    st.subheader('Bollinger Bands (Volatility)')
    lines_chart2 = px.line(df, x='Date', y=['Close', 'UpperBand', 'MiddleBand', 'LowerBand'],
                           color_discrete_map={'Close': 'goldenrod', 'UpperBand': 'blue', 'MiddleBand': 'green',
                                               'LowerBand': 'purple'})
    st.plotly_chart(lines_chart2)
    st.write('BB   Buy Signal    : ', BBbuyDate)
    st.write('BB   Sell Signal   : ', BBsellDate)
# BIAS Chart
    lines_chart7 = px.line(df, x=df['Date'],y=[df['BIAS6'],df['BIAS12'],df['BIAS24']])
    st.subheader('BIAS (Volatility)')
    st.plotly_chart(lines_chart7)
    st.write('BIAS Buy Signal  : ', BIASbuyDate)
    st.write('BIAS Sell Signal : ', BIASsellDate)

# Resistance & Support Levels
st.subheader('Resisitance & Support Levels')
st.write('Resistance Levels : ', ResistDate)
st.write('Support Levels : ', SupportDate)

# ###################################
# Prediction Models

if XGBoost :
    st.subheader('eXtreme Gradient Boosting Model')
    # Split the data into features and target
    y = df['Close']
    X = df[['Open','High','Low','Volume','EMA10','EMA50','UpperBand','MiddleBand','LowerBand','MACD','ADX','K','D','BIAS6','BIAS12','BIAS24','RSI6','RSI12','RSI24']]
    # Split the data into training and test sets
    X_train_XGB, X_test_XGB, y_train_XGB, y_test_XGB = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train an XGBoost model
    XGB_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, booster='gbtree')
    XGB_model.fit(X_train_XGB, y_train_XGB)

    # Predict on the test set
    y_pred = XGB_model.predict(X_test_XGB)
    # Add predicted prices to test data
    predicted_prices = X_test_XGB.copy()
    predicted_prices['Close'] = y_pred
    # Predict the next day close and direction
    next_day=X.tail(1)
    predicted_price = XGB_model.predict(next_day)
    predicted_price_XGB = predicted_price[0]
    st.write('  Predicts next Close Price of '+df['Date'][len(df)-1]+' is : [', str(round(predicted_price_XGB,4))+']')
    Calculate_print_metrics(y_test_XGB, y_pred)
else:
    predicted_price_XGB = 0


if New_GRU or Rel_GRU :
    st.subheader('Gated Recurrent Unit Model')
    GRU_model = Sequential()

    if New_GRU:
        GRU_model.add(GRU(units=100, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
        GRU_model.add(Dropout(0.3))
        # Second GRU layer
        GRU_model.add(GRU(units=80, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
        GRU_model.add(Dropout(0.2))
        # Third GRU layer
        GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
        GRU_model.add(Dropout(0.1))
        # Fourth GRU layer
        GRU_model.add(GRU(units=30, activation='tanh'))
        GRU_model.add(Dropout(0.2))
        # The output layer
        GRU_model.add(Dense(units=1))
        # Compiling the RNN
        GRU_model.compile(optimizer='adam',loss='mean_squared_error')
        # Fitting to the training set
        GRU_model.fit(X_train,y_train,epochs=50, batch_size=32, verbose=1)
        # Save the model as a h5 file
        GRU_model.save(code+"_GRU_model.h5")
    if Rel_GRU:
        # Reload the model from h5 file
        GRU_model = load_model(code+"_GRU_model.h5")

    # Making predictions
    predicted_stock_price = GRU_model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    last_prices = df['Close'][-prediction_days:].values.reshape(-1, 1)
    last_prices_scaled = scaler.transform(last_prices)
    predicted_price_scaled = GRU_model.predict(last_prices_scaled.reshape(1,prediction_days,1))
    predicted_price_GRU = scaler.inverse_transform(predicted_price_scaled)
    st.write("Predicts next Close Price of " + df['Date'][len(df)-1] + " is : ", str(predicted_price_GRU[0]))
    Calculate_print_metrics(test_set, predicted_stock_price)


if New_LSTM or Rel_LSTM :
    st.subheader('Long Short Term Memory Model')
    LSTM_model = Sequential()

    if New_LSTM :
        # First LSTM layer with Dropout regularisation
        LSTM_model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1],1)))
        LSTM_model.add(Dropout(0.3))
        LSTM_model.add(LSTM(units=80, return_sequences=True))
        LSTM_model.add(Dropout(0.1))
        LSTM_model.add(LSTM(units=50, return_sequences=True))
        LSTM_model.add(Dropout(0.2))
        LSTM_model.add(LSTM(units=30))
        LSTM_model.add(Dropout(0.3))
        LSTM_model.add(Dense(units=1))
        LSTM_model.compile(optimizer='adam',loss='mean_squared_error')
        LSTM_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
        # Save the model as a h5 file
        LSTM_model.save(code+"_LSTM_model.h5")

    if Rel_LSTM :
        # reload the model from h5 file
        LSTM_model = load_model(code+"_LSTM_model.h5")

    # Making predictions
    predicted_stock_price = LSTM_model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    last_prices = df['Close'][-prediction_days:].values.reshape(-1,1)
    last_prices_scaled = scaler.transform(last_prices)
    predicted_price_scaled = LSTM_model.predict(last_prices_scaled.reshape(1,prediction_days,1))
    predicted_price_LSTM = scaler.inverse_transform(predicted_price_scaled)
    st.write("Predicts next Close Price of " + df['Date'][len(df)-1] + " is :", str(predicted_price_LSTM[0]))

    Calculate_print_metrics(test_set, predicted_stock_price)


if New_LSTM_AM or Rel_LSTM_AM:
    st.subheader('Long Short Term Memory - Attention Mechanism Model')
    LSTM_AM_model = Sequential()

    if New_LSTM_AM :
        # Adding LSTM layers with return_sequences=True
        LSTM_AM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        LSTM_AM_model.add(LSTM(units=50, return_sequences=True))
        # Adding self-attention mechanism
        # The attention mechanism
        attention = AdditiveAttention(name='attention_weight')
        # Permute and reshape for compatibility
        LSTM_AM_model.add(Permute((2, 1)))
        LSTM_AM_model.add(Reshape((-1, X_train.shape[1])))
        attention_result = attention([LSTM_AM_model.output, LSTM_AM_model.output])
        multiply_layer = Multiply()([LSTM_AM_model.output, attention_result])
        # Return to original shape
        LSTM_AM_model.add(Permute((2, 1)))
        LSTM_AM_model.add(Reshape((-1, 50)))
        # Adding a Flatten layer before the final Dense layer
        LSTM_AM_model.add(tf.keras.layers.Flatten())
        # Final Dense layer
        LSTM_AM_model.add(Dense(1))
        # Adding Dropout and Batch Normalization
        LSTM_AM_model.add(Dropout(0.2))
        LSTM_AM_model.add(BatchNormalization())
        # Compile the model
        LSTM_AM_model.compile(optimizer='adam', loss='mean_squared_error')
        LSTM_AM_model.summary()
        # Train the model
        # Assuming X_train and y_train are already defined and preprocessed
        history = LSTM_AM_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        history = LSTM_AM_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping],
                        verbose=1)

        # save the model as a h5 file
        LSTM_AM_model.save(code+"_LSTM_AM_model.h5")

    if Rel_LSTM_AM :
        # reload the model from h5 file
        LSTM_AM_model = load_model(code+"_LSTM_AM_model.h5")

    # Making predictions
    predicted_stock_price = LSTM_AM_model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    last_prices = df['Close'][-50:].values.reshape(-1, 1)
    last_prices_scaled = scaler.fit_transform(last_prices)
    predicted_price_scaled = LSTM_AM_model.predict(last_prices_scaled.reshape(1, 50, 1))
    predicted_price_LSTM_AM = scaler.inverse_transform(predicted_price_scaled)
    st.write("Predicts next Close Price of " + df['Date'][len(df) - 1] + " is :", str(predicted_price_LSTM_AM[0]))

    Calculate_print_metrics(test_set, predicted_stock_price)


if New_LSTM_FEAT or Rel_LSTM_FEAT:
    st.subheader('Long Short Term Memory - Features (Tech. Indicators) Model')
    LSTM_FEAT_model = Sequential()

    # List of considered Features
    FEATURES = ['High', 'Low', 'Open', 'EMA10', 'EMA50', 'UpperBand', 'LowerBand']
    # Shift the timeframe by 50days
    df_features = df.iloc[50:].copy()
    # Filter the data to the list of FEATURES
    data_filtered_ext = df_features[FEATURES].copy()
    # create a copy of the data
    dfs = data_filtered_ext.copy()
    # Transform the data by scaling each feature to a range between 0 and 1
    scaler = RobustScaler()
    np_data = scaler.fit_transform(dfs)  # np_data_unscaled
    # Creating a separate scaler that works on a single column for scaling predictions
    scaler_pred = RobustScaler()
    df_Close = pd.DataFrame(df_features['Close'])
    np_Close_scaled = scaler_pred.fit_transform(df_Close)
    # Set the prediction days - this is the timeframe used to make a single prediction
    prediction_days = 50  # = number of neurons in the first layer of the neural network
    # Split the training data into train and train data sets
    # Create the training and test data
    train_data = np_data[:length_train, :]
    test_data = np_data[length_train - prediction_days:, :]
    # The RNN needs data with the format of [samples, time steps, features]
    # Here, we create N samples, prediction_days time steps per sample, and 7 features
    def partition_dataset(prediction_days, data):
        X, y = [], []
        data_len = data.shape[0]
        for i in range(prediction_days, data_len):
            X.append(data[i - prediction_days:i, :])  # contains prediction_days values 0-prediction_days * columns
            y.append(data[i, 0])  # contains the prediction values for validation,  for single-step prediction
        # Convert the X and y to numpy arrays
        X = np.array(X)
        y = np.array(y)
        return X, y
    # Generate training data and test data
    X_train, y_train = partition_dataset(prediction_days, train_data)
    X_test, y_test = partition_dataset(prediction_days, test_data)

    if New_LSTM_FEAT :
        # Configure the Neural Network Model with n Neurons - inputshape = t Timestamps x f Features
        n_neurons = X_train.shape[1] * X_train.shape[2]
        LSTM_FEAT_model.add(LSTM(n_neurons, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        # model.add(Dropout(0.1))
        LSTM_FEAT_model.add(LSTM(n_neurons, return_sequences=True))
        # model.add(Dropout(0.1))
        LSTM_FEAT_model.add(LSTM(n_neurons, return_sequences=False))
        LSTM_FEAT_model.add(Dense(32))
        LSTM_FEAT_model.add(Dense(1, activation='relu'))
        # Configure the Model
        optimizer = 'adam'; loss = 'mean_squared_error'; epochs = 50; batch_size = 32; patience = 8
        # uncomment to customize the learning rate
        learn_rate = "standard"  # 0.05
        # Compile and Training the model
        LSTM_FEAT_model.compile(optimizer=optimizer, loss=loss)
        early_stop = EarlyStopping(monitor='loss', patience=patience)
        history = LSTM_FEAT_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[early_stop],
                                      shuffle=True, validation_data=(X_test, y_test), verbose=1)
        # save the model as a h5 file
        LSTM_FEAT_model.save(code+"_LSTM_FEAT_model.h5")

    if Rel_LSTM_FEAT :
        # reload the model from h5 file
        LSTM_FEAT_model = load_model(code+"_LSTM_FEAT_model.h5")

    # Get the predicted values
    y_pred_scaled = LSTM_FEAT_model.predict(X_test)
    # Unscale the predicted values
    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

    last_prices = test_data[-50:]
    predicted_price_LSTM_FEAT_scaled = LSTM_FEAT_model.predict(last_prices.reshape(1,50,7))
    predicted_price_LSTM_FEAT = scaler_pred.inverse_transform(predicted_price_LSTM_FEAT_scaled)
    st.write("Predicts next Close Price of " + df['Date'][len(df) - 1] + " is :", str(predicted_price_LSTM_FEAT[0]))

    Calculate_print_metrics(y_test_unscaled, y_pred)

if Last_Close_Price <= predicted_price_XGB and Last_Close_Price <= predicted_price_GRU and Last_Close_Price <= predicted_price_LSTM and Last_Close_Price <= predicted_price_LSTM_AM and Last_Close_Price <= predicted_price_LSTM_FEAT:
    st.subheader("{ Note : Selected model(s) predict(s) next Close Price(s) would go up, recommend to 'Buy'. }")
else:
    st.subheader("{ Note : Selected model(s) predict(s) next Close Price(s) would not go up, recommend to 'sell or 'not Buy'. }")
st.text("")

# ###################################
st.subheader('Full Stock Data')
st.dataframe(df)
