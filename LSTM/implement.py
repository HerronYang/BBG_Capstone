from config import *
from train import LSTMModel, dataPreprocessing

def implementModel(tic):
    # Load the full original dataset to retrieve dates
    original_data = pd.read_csv(r'data0_200\train_data_0_20.csv', low_memory=False)
    
    # Define sequence length
    sequence_length = 30

    # Split the data by the date and remove the first 30 rows for both train and test sets
    train_data = original_data.loc[original_data['date'] <= '2023-10-01'].iloc[sequence_length:].reset_index(drop=True)
    test_data = original_data.loc[original_data['date'] > '2023-10-01'].iloc[sequence_length:].reset_index(drop=True)

    # Load processed data from dataPreprocessing function
    data = dataPreprocessing(tic)
    _, test, X_train, y_train, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, train_loader, test_loader, scaler_target = data

    # Remove the first 'sequence_length' rows from training and testing data
    X_train, y_train = X_train[sequence_length:], y_train[sequence_length:]
    X_train_tensor, y_train_tensor = X_train_tensor[sequence_length:], y_train_tensor[sequence_length:]
    X_test_tensor, y_test_tensor = X_test_tensor[sequence_length:], y_test_tensor[sequence_length:]
    
    # Extract dates after trimming
    train_dates = train_data['date'].reset_index(drop=True)
    test_dates = test_data['date'].reset_index(drop=True)

    # Initialize model
    input_dim = X_train.shape[2]
    hidden_dim = 32
    num_layers = 1
    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    state_dict = torch.load('models/' + 'model_' + tic + '.pth')
    model.load_state_dict(state_dict)
    
    # Calculate MAE loss for training data
    with torch.no_grad():
        X_train_pred = model(X_train_tensor)
        train_mae_loss = torch.mean(torch.abs(X_train_pred - y_train_tensor), dim=1).numpy()
    
    # Set threshold based on 99th percentile of training MAE loss
    threshold = np.percentile(train_mae_loss, 99)
    print(f'Reconstruction error threshold: {threshold}')

    # Create train_score_df
    train_score_df = pd.DataFrame(y_train).reset_index(drop=True)
    train_score_df['loss'] = train_mae_loss
    train_score_df['threshold'] = threshold
    train_score_df['anomaly'] = train_score_df['loss'] > train_score_df['threshold']
    train_score_df['close'] = scaler_target.inverse_transform(y_train.reshape(-1, 1))
    train_score_df['date'] = pd.to_datetime(train_dates)
    train_score_df['tic'] = tic
    train_score_df['set'] = 'train'

    # Calculate MAE loss for test data
    with torch.no_grad():
        X_test_pred = model(X_test_tensor)
        test_mae_loss = torch.mean(torch.abs(X_test_pred - y_test_tensor), dim=1).numpy()

    # Create test_score_df by aligning it exactly with the length of test_mae_loss
    test_score_df = pd.DataFrame(test.iloc[sequence_length:sequence_length + len(test_mae_loss)]).reset_index(drop=True)
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = threshold
    test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
    test_score_df['close'] = scaler_target.inverse_transform(test_score_df['close'].values.reshape(-1, 1))
    test_score_df['date'] = pd.to_datetime(test_dates[:len(test_mae_loss)])
    test_score_df['tic'] = tic
    test_score_df['set'] = 'test'

    # Combine train and test DataFrames
    combined_df = pd.concat([train_score_df, test_score_df], ignore_index=True)
    # Select only date, tic, and anomaly columns for the final output
    final_df = combined_df[['date', 'tic', 'anomaly']]

    print(final_df.head())
    anomaly_count = final_df['anomaly'].sum()
    print(f"Number of anomalies: {anomaly_count}")

    final_df.to_csv('results/result_' + tic + '.csv')

    return final_df

if __name__ == '__main__':
    data = pd.read_csv('data0_200/train_data_0_20.csv')
    tic = data.tic.unique()
    for i in tqdm(tic):
        implementModel(i)

'''
    # Plot test MAE loss and threshold
    plt.figure(figsize=(10, 6))
    plt.plot(test_score_df['date'], test_score_df['loss'], label='Test loss')
    plt.plot(test_score_df['date'], test_score_df['threshold'], label='Threshold', linestyle='--')
    plt.title('Test loss vs. Threshold')
    plt.xlabel('date')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Visualize anomalies on the actual close price
    plt.figure(figsize=(10, 6))
    plt.plot(test_score_df['date'], test_score_df['close'], label='Close price')
    plt.scatter(test_score_df.loc[test_score_df['anomaly'], 'date'], 
                test_score_df.loc[test_score_df['anomaly'], 'close'], 
                color='red', label='Anomaly', marker='o')
    plt.title('Detected anomalies in close price')
    plt.xlabel('date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
'''