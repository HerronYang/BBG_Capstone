from config import *

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)  # Dropout to reduce overfitting
        self.fc = nn.Linear(hidden_dim, 1)  # Fully connected layer for prediction

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]  # Take the hidden state of the last time step
        last_hidden_state = self.dropout(last_hidden_state)  # Apply dropout
        output = self.fc(last_hidden_state)  # Get the final prediction
        return output

# Function to create sequences for input features (X) and target (y)
def create_sequences(data, target, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]  # Input features sequence
        sequences.append(sequence)
        targets.append(target[i + sequence_length])  # Corresponding 'close' target
    return np.array(sequences), np.array(targets)
    

def dataPreprocessing(tic):
    # 1. Data Loading
    data = pd.read_csv(r'train_data_0_20.csv')
    data = data[data.tic == tic]

    # Split into train and test datasets
    train, test = data.loc[data['date'] <= '2023-10-01'], data.loc[data['date'] > '2023-10-01']
    # 2. Data Preprocessing
    variables = ['close', 'volume', 'day', 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 
                'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix', 'turbulence']  # Only input features

    # Remove NaNs if there are any missing values in the dataset
    train = train.dropna()
    test = test.dropna()

    # Standardization of input features (variables)
    scaler = StandardScaler()
    train[variables] = scaler.fit_transform(train[variables])
    test[variables] = scaler.transform(test[variables])

    # Standardization of the target 'close' column separately (since close is the target)
    scaler_target = StandardScaler()
    train['close'] = scaler_target.fit_transform(train[['close']])
    test['close'] = scaler_target.transform(test[['close']])

    # Convert DataFrame to NumPy array for processing (only input features for X)
    train_data = train[variables].values  # Only input features
    test_data = test[variables].values  # Only input features

    # Sequence length (e.g., 30 days)
    sequence_length = 30

    # Create sequences for training and testing, with target as 'close' price
    X_train, y_train = create_sequences(train_data, train['close'].values, sequence_length)
    X_test, y_test = create_sequences(test_data, test['close'].values, sequence_length)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for correct shape
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for correct shape

    # Create DataLoader for mini-batch training
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train, test, X_train, y_train, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, train_loader, test_loader, scaler_target


def trainModel(tic):
    train, test, X_train, y_train, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, train_loader, test_loader, scaler_target = dataPreprocessing(tic)
    input_dim = X_train.shape[2]  # number of features
    hidden_dim = 32  # Reduce hidden units to 32 to avoid overfitting
    num_layers = 1  # Using 1 layer for simplicity

    # Initialize the model
    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    # 4. Define Loss Function, Optimizer, and Learning Rate Scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # L2 regularization
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  # Reduce LR every 30 epochs

    # 5. Early Stopping Parameters
    patience = 10
    best_test_loss = float('inf')
    patience_counter = 0

    # 6. Training the Model with Gradient Clipping and Early Stopping
    num_epochs = 500
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0

        for X_batch, y_batch in train_loader:
            # Forward pass
            output = model(X_batch)
            loss = criterion(output, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Apply gradient clipping
            optimizer.step()

            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))

        # Evaluation on test data
        model.eval()
        with torch.no_grad():
            epoch_test_loss = 0
            for X_batch, y_batch in test_loader:
                test_output = model(X_batch)
                test_loss = criterion(test_output, y_batch)
                epoch_test_loss += test_loss.item()

            test_losses.append(epoch_test_loss / len(test_loader))

        # Adjust learning rate
        scheduler.step()

        # Early stopping check
        if test_losses[-1] < best_test_loss:
            best_test_loss = test_losses[-1]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # if epoch % 10 == 0:
            # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

        # 7. save the model
        torch.save(model.state_dict(), 'models/model_'+tic+'.pth')

if __name__ == '__main__':
    data = pd.read_csv('train_data_0_20.csv')
    tic = data.tic.unique()
    for i in tqdm(tic):
        trainModel(i)