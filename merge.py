import pandas as pd
import os 

# Load the CSV files
final_merged_df = pd.read_csv('df_final_merged_renewed.csv')


# Initialize the 'LSTM_anomaly' column in the final merged DataFrame with default values
final_merged_df['LSTM_anomaly'] = False


# Specify the directory containing the result files
results_folder = 'results'

# Iterate over each file in the results folder
for filename in os.listdir(results_folder):
    if filename.startswith('result_') and filename.endswith('.csv'):
        # Load the result file and drop any unnecessary columns
        result_df = pd.read_csv(os.path.join(results_folder, filename)).drop(columns=['Unnamed: 0'], errors='ignore')
        
        # Ensure 'LSTM_anomaly' column exists and rename if needed
        if 'anomaly' in result_df.columns:
            result_df.rename(columns={'anomaly': 'LSTM_anomaly'}, inplace=True)
        
        # Keep only necessary columns for merging
        result_df = result_df[['date', 'tic', 'LSTM_anomaly']]
        
        # Merge the result file with the main DataFrame, updating 'LSTM_anomaly' where True
        final_merged_df = final_merged_df.merge(result_df, on=['date', 'tic'], how='left', suffixes=('', '_new'))
        
        # Check if 'LSTM_anomaly_new' column was created in the merge, and update if so
        if 'LSTM_anomaly_new' in final_merged_df.columns:
            # Update 'LSTM_anomaly' in final_merged_df where 'LSTM_anomaly_new' is True
            final_merged_df['LSTM_anomaly'] = final_merged_df['LSTM_anomaly'] | final_merged_df['LSTM_anomaly_new']
            
            # Drop the temporary 'LSTM_anomaly_new' column
            final_merged_df.drop(columns=['LSTM_anomaly_new'], inplace=True)
final_merged_df['LSTM_anomaly'] = final_merged_df['LSTM_anomaly'].astype(int)
# Display the first few rows of the updated final merged DataFrame
print(final_merged_df.head())

true_count = final_merged_df['LSTM_anomaly'].sum()
print(f"Number of True values in LSTM_anomaly: {true_count}")

# Save the updated final merged DataFrame to a new CSV file
final_merged_df.to_csv('final_merged.csv')

