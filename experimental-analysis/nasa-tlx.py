import tkinter as tk
from tkinter import messagebox
import csv
import os

# Initialize participant count
participant_count = 1

# Check if CSV file exists and initialize participant_count
csv_file = 'nasa_tlx_data.csv'
if os.path.exists(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        if rows:
            # Update participant_count based on the last participant number in the file
            participant_count = int(rows[-1][0]) + 1
else:
    # If file does not exist, create it with headers
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow([
            "Participant Number", 
            "Mental Demand", 
            "Physical Demand", 
            "Temporal Demand", 
            "Performance", 
            "Effort", 
            "Frustration", 
            "Overall Score"
        ])

# Function to save the entered data to a CSV file
def save_to_csv():
    global participant_count
    try:
        # Collecting input values
        mental_demand = int(mental_demand_entry.get())
        physical_demand = int(physical_demand_entry.get())
        temporal_demand = int(temporal_demand_entry.get())
        performance = int(performance_entry.get())
        effort = int(effort_entry.get())
        frustration = int(frustration_entry.get())
        
        # Validate input values
        if not all(0 <= x <= 100 for x in [mental_demand, physical_demand, temporal_demand, performance, effort, frustration]):
            raise ValueError("All values must be between 0 and 100")
        
        # Calculate overall score
        overall_score = (mental_demand + physical_demand + temporal_demand + performance + effort + frustration) / 6
        
        # Data to write
        data = [
            participant_count,
            mental_demand,
            physical_demand,
            temporal_demand,
            performance,
            effort,
            frustration,
            overall_score
        ]
        
        # Write data to CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        
        # Show success message
        messagebox.showinfo("Success", "Data saved successfully!")
        
        # Increment participant count for the next entry
        participant_count += 1
        
        # Clear inputs
        clear_inputs()
        
    except ValueError as e:
        messagebox.showerror("Error", str(e))

# Function to clear input fields
def clear_inputs():
    mental_demand_entry.delete(0, tk.END)
    physical_demand_entry.delete(0, tk.END)
    temporal_demand_entry.delete(0, tk.END)
    performance_entry.delete(0, tk.END)
    effort_entry.delete(0, tk.END)
    frustration_entry.delete(0, tk.END)

# Initialize the main application window
app = tk.Tk()
app.title("NASA TLX Survey")

# Create labels, questions, and entry fields for each NASA-TLX subscale
tk.Label(app, text="Mental Demand (0-100):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
mental_demand_entry = tk.Entry(app)
mental_demand_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")
tk.Label(app, text="How mentally demanding was the task?").grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")

tk.Label(app, text="Physical Demand (0-100):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
physical_demand_entry = tk.Entry(app)
physical_demand_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
tk.Label(app, text="How physically demanding was the task?").grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="w")

tk.Label(app, text="Temporal Demand (0-100):").grid(row=4, column=0, padx=10, pady=5, sticky="w")
temporal_demand_entry = tk.Entry(app)
temporal_demand_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")
tk.Label(app, text="How hurried or rushed was the pace of the task?").grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="w")

tk.Label(app, text="Performance (0-100):").grid(row=6, column=0, padx=10, pady=5, sticky="w")
performance_entry = tk.Entry(app)
performance_entry.grid(row=6, column=1, padx=10, pady=5, sticky="w")
tk.Label(app, text="How successful were you in accomplishing what you were asked to do?").grid(row=7, column=0, columnspan=2, padx=10, pady=5, sticky="w")

tk.Label(app, text="Effort (0-100):").grid(row=8, column=0, padx=10, pady=5, sticky="w")
effort_entry = tk.Entry(app)
effort_entry.grid(row=8, column=1, padx=10, pady=5, sticky="w")
tk.Label(app, text="How hard did you have to work to accomplish your level of performance?").grid(row=9, column=0, columnspan=2, padx=10, pady=5, sticky="w")

tk.Label(app, text="Frustration (0-100):").grid(row=10, column=0, padx=10, pady=5, sticky="w")
frustration_entry = tk.Entry(app)
frustration_entry.grid(row=10, column=1, padx=10, pady=5, sticky="w")
tk.Label(app, text="How insecure, discouraged, irritated, stressed, and annoyed were you?").grid(row=11, column=0, columnspan=2, padx=10, pady=5, sticky="w")

# Button to submit the scores
submit_button = tk.Button(app, text="Submit", command=save_to_csv)
submit_button.grid(row=12, column=0, columnspan=2, pady=10, padx=50)

# Run the application
app.mainloop()
