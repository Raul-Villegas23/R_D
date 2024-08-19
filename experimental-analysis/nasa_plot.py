import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
csv_file = 'merged_nasa_tlx_data.csv'
data = pd.read_csv(csv_file)

# Display the first few rows of the dataframe to verify the contents
print(data.head())

# Plotting each dimension vs Delay
def plot_dimension(data, dimension, ax):
    ax.bar(data['Delay'], data[dimension], color='blue', alpha=0.7)
    ax.set_title(f'{dimension} Scores vs Delay')
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 100)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Create subplots for each dimension and overall score
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle('NASA TLX Results vs Time Delays')

# List of dimensions to plot
dimensions = [
    "Mental Demand", 
    "Physical Demand", 
    "Temporal Demand", 
    "Performance", 
    "Effort", 
    "Frustration"
]

# Plot each dimension in its subplot
for i, dimension in enumerate(dimensions):
    plot_dimension(data, dimension, axs[i // 2, i % 2])

# Plot Overall Score separately
axs[2, 1].bar(data['Delay'], data['Overall Score'], color='green', alpha=0.7)
axs[2, 1].set_title('Overall Scores vs Delay')
axs[2, 1].set_xlabel('Delay (ms)')
axs[2, 1].set_ylabel('Score')
axs[2, 1].set_ylim(0, 100)
axs[2, 1].grid(True, which='both', linestyle='--', linewidth=0.5)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Plotting a single graph to see everything vs Delay
plt.figure(figsize=(14, 8))
for dimension in dimensions:
    plt.plot(data['Delay'], data[dimension], marker='o', label=dimension)

plt.plot(data['Delay'], data['Overall Score'], marker='o', color='black', linestyle='--', linewidth=2, label='Overall Score')
plt.title('NASA TLX Scores Across Delays')
plt.xlabel('Delay (ms)')
plt.ylabel('Score')
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# Plotting a pie chart for average contribution
average_scores = data[dimensions].mean()
plt.figure(figsize=(8, 8))
plt.pie(average_scores, labels=dimensions, autopct='%1.1f%%', startangle=140)
plt.title('Average Contribution of Each Dimension')
plt.show()

# Plotting a box plot for each dimension
plt.figure(figsize=(12, 8))
data.boxplot(column=dimensions + ['Overall Score'], grid=False)
plt.title('Box Plot of NASA TLX Dimensions')
plt.ylabel('Score')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.show()
