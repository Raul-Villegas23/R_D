import pandas as pd

# Load the NASA TLX data
nasa_tlx_file = 'nasa_tlx_data.csv'
nasa_tlx_data = pd.read_csv(nasa_tlx_file)

# Load the participants delays data
participants_delays_file = 'participants_delays.csv'
participants_delays_data = pd.read_csv(participants_delays_file)

# Display the first few rows of both dataframes to verify the contents
print("NASA TLX Data:")
print(nasa_tlx_data.head())

print("\nParticipants Delays Data:")
print(participants_delays_data.head())

# Add an index to each DataFrame based on the row number
nasa_tlx_data['Row Index'] = nasa_tlx_data.index
participants_delays_data['Row Index'] = participants_delays_data.index

# Merge the two DataFrames based on the Row Index
merged_data = pd.merge(nasa_tlx_data, participants_delays_data, how='left',
                       on='Row Index')

# Drop the Row Index column
merged_data.drop(columns=['Row Index'], inplace=True)

# Rename columns for clarity
merged_data.rename(columns={'Delays': 'Delay'}, inplace=True)

# Display the merged dataframe to verify the result
print("\nMerged Data:")
print(merged_data.head())

# Save the merged data to a new CSV file
merged_file = 'merged_nasa_tlx_data.csv'
merged_data.to_csv(merged_file, index=False)
