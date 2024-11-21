import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getAnomalyWithSensitivity(df_path, tic, Sensitivity):
    # Read data and group tic
    df = pd.read_csv(df_path)
    df_tic = df[df['tic'] == tic]
    df_tic['date'] = pd.to_datetime(df_tic['date'])

    # Get train and test data
    df_tic_train = df_tic[df_tic['date'] <= pd.Timestamp('2023-10-01')]
    df_tic_test = df_tic[df_tic['date'] > pd.Timestamp('2023-10-01')]


    # Get threshold
    train_threshold = np.percentile(df_tic_train['anomaly'], Sensitivity)
    test_threshold = np.percentile(df_tic_test['anomaly'], Sensitivity)

    # anomaly
    df_tic_train['LSTM_anomaly'] = df_tic_train['anomaly'] > train_threshold
    df_tic_test['LSTM_anomaly'] = df_tic_test['anomaly'] > test_threshold

    # Merge data
    combined_df = pd.concat([df_tic_train, df_tic_test])
    df_tic = df_tic.merge(combined_df[['date', 'LSTM_anomaly']], how='left', on='date')

    return df_tic


if __name__ == '__main__':
    def plot_anomalies(tickers):
        # Define the grid size based on the number of tickers
        rows, cols = 5, 4  # Adjust based on layout preferences (e.g., 5x4 for 20 slots, leaving 3 empty)
        fig, axs = plt.subplots(rows, cols, figsize=(20, 15))
        fig.subplots_adjust(hspace=0.5, wspace=0.3)

        for i, tic in enumerate(tickers):
            # Get the data for the current ticker
            data = getAnomalyWithSensitivity('final_merged.csv', tic, 95)

            # Calculate the row and column indices for this subplot
            row, col = divmod(i, cols)

            # Plot the closing price
            axs[row, col].plot(data['date'], data['close'], label='Close Price', color='blue')

            # Mark anomaly points
            anomaly = data[data['LSTM_anomaly'] == 1]
            axs[row, col].scatter(anomaly['date'], anomaly['close'], color='red', label='Anomaly')

            # Set the title for each subplot
            axs[row, col].set_title(f'{tic} Anomaly Detection')
            axs[row, col].set_xlabel('Date')
            axs[row, col].set_ylabel('Close Price')

        # Turn off any unused subplots if there are empty spaces
        for j in range(i + 1, rows * cols):
            fig.delaxes(axs.flatten()[j])

        plt.show()


    tickers = pd.read_csv('final_merged.csv').tic.unique().tolist()
    plot_anomalies(tickers)
