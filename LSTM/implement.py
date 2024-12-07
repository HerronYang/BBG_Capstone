from config import *
from train import LSTMModel, dataPreprocessing
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import numpy as np


def compute_anomaly_probabilities(loss_values):
    """
    Convert a list of loss values into anomaly probabilities using z-score standardization
    and the survival function, similar to a scaler.

    Parameters:
    - loss_values (list or np.array): List or array of loss values.

    Returns:
    - np.array: Probabilities scaled between 0 and 1.
    """
    # Convert the list to a numpy array for easier computation
    loss_values = np.array(loss_values)

    # Compute mean and standard deviation
    mean_loss = loss_values.mean()
    std_loss = loss_values.std()

    # Handle edge case: if all values are the same
    if std_loss == 0:
        return np.full_like(loss_values, 0.5, dtype=float)  # Default to neutral probability

    # Standardize loss values (z-score)
    z_scores = (loss_values - mean_loss) / std_loss

    # Convert z-scores to probabilities using the survival function
    probabilities = 1 - norm.sf(z_scores)

    return probabilities

def implementModel(tic):
    # Load the full original dataset to retrieve dates
    original_data = pd.read_csv('train_data_0_20.csv0', low_memory=False)
    
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
    
def getAnomalies(tic):
    scaler = MinMaxScaler()
    original_data = pd.read_csv('train_data_0_20.csv', low_memory=False)
    
    # Define sequence length
    sequence_length = 30

    # Split the data by the date and remove the first 30 rows for both train and test sets
    train_data = original_data.loc[original_data['date'] <= '2023-10-01'].iloc[sequence_length:].reset_index(drop=True)
    test_data = original_data.loc[original_data['date'] > '2023-10-01'].iloc[sequence_length:].reset_index(drop=True)

    # Load processed data from dataPreprocessing function
    data = dataPreprocessing(tic)
    train, test, X_train, y_train, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, train_loader, test_loader, scaler_target = data

    input_dim = X_train.shape[2]
    hidden_dim = 32
    num_layers = 1
    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    state_dict = torch.load('models/' + 'model_' + tic + '.pth')
    model.load_state_dict(state_dict)

    # Train data predictions and anomaly detection
    with torch.no_grad():
        X_train_pred = model(X_train_tensor)
        train_mae_loss = torch.mean(torch.abs(X_train_pred - y_train_tensor), dim=1).numpy()

    train_score_df = pd.DataFrame(train[sequence_length:]).reset_index(drop=True)
    train_score_df['loss'] = train_mae_loss
    train_score_df['anomaly'] = compute_anomaly_probabilities(train_score_df[['loss']])

    # Inverse transform the close prices back to their original scale
    train_score_df['close'] = scaler_target.inverse_transform(train[sequence_length:]['close'].values.reshape(-1, 1))

    # Test data predictions and anomaly detection
    with torch.no_grad():
        X_test_pred = model(X_test_tensor)
        test_mae_loss = torch.mean(torch.abs(X_test_pred - y_test_tensor), dim=1).numpy()

    test_score_df = pd.DataFrame(test[sequence_length:]).reset_index(drop=True)
    test_score_df['loss'] = test_mae_loss
    test_score_df['anomaly'] = compute_anomaly_probabilities(test_score_df[['loss']])

    # Inverse transform the close prices back to their original scale
    test_score_df['close'] = scaler_target.inverse_transform(test[sequence_length:]['close'].values.reshape(-1, 1))

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
    data = pd.read_csv('train_data_0_20.csv', low_memory=False)
    tic = data.tic.unique()
    for i in tqdm(tic):
        getAnomalies(i)