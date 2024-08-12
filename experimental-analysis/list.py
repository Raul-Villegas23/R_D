import pandas as pd
import numpy as np

# Sample data loading (replace the file name with your actual data file path)
file_path = 'Experiments_chart_list.csv'
df = pd.read_csv(file_path, delimiter=';', decimal=',')   # Assuming your numbers use commas as decimal separators
print(df)


# Function to validate and ensure unique 'Delays' and 'Lag Speed' for each participant
def validate_entries(df):
    print("Starting validation...")
    for participant in df['Participants'].unique():
        print(f"Processing {participant}...")
        sub_df = df[df['Participants'] == participant]
        for column in ['Delays', 'Lag Speed']:
            iteration = 0
            while sub_df[column].duplicated().any():
                print(f"Resolving duplicates in {column} for {participant}...")
                duplicated = sub_df[sub_df[column].duplicated()].index
                for index in duplicated:
                    new_value = generate_new_value(sub_df, column)
                    df.at[index, column] = new_value
                iteration += 1
                if iteration > 10:  # Break loop if too many iterations
                    print("Too many iterations, breaking...")
                    break
            sub_df = df[df['Participants'] == participant]  # Update sub_df after changes
    print("Validation completed.")
    return df

def generate_new_value(sub_df, column):
    existing_values = set(sub_df[column])
    if column == 'Delays':
        possible_range = range(0, 2000)  # Adjust the range if needed
    else:  # Assuming 'Lag Speed' needs decimal values
        possible_range = [x * 0.01 for x in range(1, 2000)]  # Adjust the range and step if needed
    new_value_candidates = [value for value in possible_range if value not in existing_values]
    if not new_value_candidates:  # If no candidates are available, just increment the max value
        return sub_df[column].max() + (0.01 if column == 'Lag Speed' else 1)
    return np.random.choice(new_value_candidates)

# Validate the DataFrame
validated_df = validate_entries(df)
print(validated_df)

# Optionally, save the validated DataFrame back to CSV
validated_df.to_csv('Experiments_chart_list2.csv', index=False)
