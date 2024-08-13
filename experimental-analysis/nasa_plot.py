import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
csv_file = 'nasa_tlx_data.csv'
data = pd.read_csv(csv_file)

# Display the first few rows of the dataframe to verify the contents
print(data.head())

# Plotting each dimension in separate bar charts
def plot_dimension(data, dimension, ax):
    ax.bar(data['Participant Number'], data[dimension], color='blue', alpha=0.7)
    ax.set_title(f'{dimension} Scores')
    ax.set_xlabel('Participant Number')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 100)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Create subplots for each dimension and overall score
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle('NASA TLX Results')

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
axs[2, 1].bar(data['Participant Number'], data['Overall Score'], color='green', alpha=0.7)
axs[2, 1].set_title('Overall Scores')
axs[2, 1].set_xlabel('Participant Number')
axs[2, 1].set_ylabel('Score')
axs[2, 1].set_ylim(0, 100)
axs[2, 1].grid(True, which='both', linestyle='--', linewidth=0.5)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Plotting a single graph to see everything
plt.figure(figsize=(14, 8))
for dimension in dimensions:
    plt.plot(data['Participant Number'], data[dimension], marker='o', label=dimension)

plt.plot(data['Participant Number'], data['Overall Score'], marker='o', color='black', linestyle='--', linewidth=2, label='Overall Score')
plt.title('NASA TLX Scores Across Participants')
plt.xlabel('Participant Number')
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
