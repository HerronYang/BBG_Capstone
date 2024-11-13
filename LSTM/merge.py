import pandas as pd
import os 

# Load the CSV files
final_merged_df = pd.read_csv('df_final_merged_renewed.csv')


# Initialize the 'LSTM_anomaly' column in the final merged DataFrame with default values
final_merged_df['LSTM_anomaly'] = False


# Specify the directory containing the result files
results_folder = 'results'

result_dfs = [] 

# Iterate over each file in the results folder
for filename in os.listdir(results_folder):
    if filename.startswith('result_') and filename.endswith('.csv'):
        # Load the result file and drop any unnecessary columns
        result_df = pd.read_csv(os.path.join(results_folder, filename)).drop(columns=['Unnamed: 0'], errors='ignore')
        result_dfs.append(result_df)

# Ensure date formatting is consistent in each result_df
for result_df in result_dfs:
    result_df['date'] = pd.to_datetime(result_df['date'], format='%Y/%m/%d')

# Concatenate all result DataFrames into one
combined_result_df = pd.concat(result_dfs, ignore_index=True)

# Drop duplicate rows if they exist
combined_result_df = combined_result_df.drop_duplicates(subset=['date', 'tic'])

print(combined_result_df)

# Merge combined_result_df with final_merged_df
final_merged_df['date'] = pd.to_datetime(final_merged_df['date'], format='%Y-%m-%d')  # Ensure date formatting in final_merged_df
final_merged_df = final_merged_df.merge(combined_result_df[['date', 'tic', 'anomaly']], on=['date', 'tic'], how='left')


print(final_merged_df['anomaly'].value_counts())

# Display the first few rows to verify
print(final_merged_df.head())

final_merged_df['anomaly'] = final_merged_df['anomaly'].apply(lambda x: 1 if x is True else 0)
final_merged_df.drop('LSTM_anomaly', axis=1, inplace=True)
print(final_merged_df.head())
final_merged_df.to_csv('final_merged.csv')

