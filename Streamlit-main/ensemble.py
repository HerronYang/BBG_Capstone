import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime, timedelta
from streamlit_date_picker import date_range_picker, date_picker, PickerType
# matplotlib.use('TkAgg')

st.title("Anomaly Visualization of the ensemble model")
st.write('Bloomberg capstone group Bravo:')
st.write("Xinran Cheng, Zhaoyang Hong, Qi Wu, Haoran Yang, Cleve He")

st.write("""\n
This is a visualization of the detected anomalies by the ensemble model combining all models, including statistical, DBSCAN, Isolation forest, One-class SVM, Autoencoder, and LSTM.\n 
Available stock universe is the top 17 of the 200 least liquid stocks in Russell 2000. 
To view the labelled anomalies, choose one ticker from the selection bar and input the desired number of anomalies given by each model .
Time horizon ranges from 2015-10-20 to 2024-08-29 and can be adjusted through the date range picker. 
""")

data = pd.read_csv("df_final_prob_renewed.csv", parse_dates=['date'])

# data = data[['date', 'tic', 'close', 'volume',
#        'DBSCAN_Anomaly_Probability','IsolationForest_Anomaly_Probability','OCSVM_Anomaly_Probability','Autoencoder_Anomaly_Probability','LSTM_Anomaly_Probability','stat_Anomaly_Probability']]

ticker_list = data['tic'].unique()


# ticker = 'ARL'

ticker = st.selectbox(
    "Select a ticker",
    ticker_list,
    placeholder='Select...'
)


anom_num=st.number_input(
    "Input the desired number of anomalies ", min_value=1, value=50, step=1, placeholder="Type an integer..."
)
st.write(f"The number of anomalies allowed is {int(anom_num):d}.")

default_start= datetime(2015, 10, 20)
default_end=datetime(2024, 8, 29)
dstart, dend = default_start,default_end
available_datas = []
date_range = default_start
while date_range <= default_end:
    available_datas.append(date_range)
    date_range += timedelta(days=1)

date_range_string = date_range_picker(picker_type=PickerType.date,
                                      start=default_start, end=default_end,
                                      available_dates=available_datas,
                                      key='available_date_range_picker',)
if date_range_string:
    dstart, dend = date_range_string
    st.write(f"Date Range Picker [{dstart}, {dend}]")


word_match = { # show : colname
    'DBSCAN': 'DBSCAN',
    'Isolation Forest': 'IsolationForest',
    'One-class SVM' : 'OCSVM',
    'Autoencoder': 'Autoencoder',
    'LSTM' : 'LSTM',
    'Statistical Model' : 'stat',
}
model_list = list(word_match.keys())
color_match = { # show : colname
    'DBSCAN': 'red',
    'Isolation Forest': 'green',
    'One-class SVM' : 'darkgoldenrod',
    'Autoencoder': 'blue',
    'LSTM' : 'darkviolet',
    'Statistical Model' : 'deeppink',
}

if ticker and anom_num and date_range_string:
    
    # Filter the data for the specified ticker
    data_tic_all = data[data['tic'] == ticker].copy()
    data_tic = data_tic_all[(data_tic_all['date'] >= dstart) & (data_tic_all['date']<= dend)].copy()
    nds=len(data_tic)

    # calculate return data
    data_tic['return'] = data_tic['close'].pct_change(fill_method=None)
    data_tic['log_volume'] = np.log(data_tic['volume']+1)

    # Plotting
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(16,32))

    # Plot the close price and volume
    ax[0].plot(data_tic['date'], data_tic['close'], color='wheat',label='Close Price')
    ax[0].set(title = f'{ticker} Price Anomalies')

    ax[1].plot(data_tic['date'], data_tic['return'], color='wheat', label='Return')
    ax[1].set(title = f'{ticker} Return Anomalies')
    
    ax[2].fill_between(data_tic['date'], 0, data_tic['log_volume'],facecolor='wheat', label='log Volume', alpha=0.8)
    ax[2].set(title=f'{ticker} Volume Anomalies')

    for model in model_list:
        modelsc= word_match[model]
        mdcolor= color_match[model]
        # Filter for anomalies with highest 'Anomaly Probability' for the desired number
        anomalies = data_tic.sort_values(by=f'{modelsc}_Anomaly_Probability', ascending=False).head(min(nds,anom_num))
        # nps1=len(anomalies)
    
        # Mark anomalies for model
        ax[0].scatter(anomalies['date'], anomalies['close'], color=mdcolor, label=f'{modelsc} Anomalies', marker='^')
        ax[1].scatter(anomalies['date'], anomalies['return'], color=mdcolor, label=f'{modelsc} Anomalies', marker='^')
        ax[2].bar(anomalies['date'], anomalies['log_volume'], color=mdcolor, label=f'{modelsc} Anomalies', width=1)
    
    
    # show legend and xlabel
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.xlabel('Date')

    
    st.pyplot(fig)
    st.write(f"{anom_num} anomalies allowed for each model within {nds} days.")
    # plt.show()
