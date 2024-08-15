import tkinter as tk
from tkinter import messagebox
import csv

# Function to save the entered data to a CSV file
def save_to_csv():
    try:
        # Collecting input values
        name = name_entry.get()
        age = age_entry.get()
        gender = gender_var.get()
        profession = profession_entry.get()
        gaming_experience = gaming_experience_var.get()

        # Validate age
        if not age.isdigit():
            raise ValueError("Age must be a number")

        # Validate that all fields are filled
        if not name or not age or not gender or not profession or not gaming_experience:
            raise ValueError("All fields must be filled out")

        # Data to write
        data = [name, age, gender, profession, gaming_experience]

        # Write data to CSV file
        with open('participants_info.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

        # Show success message
        messagebox.showinfo("Success", "Data saved successfully!")

        # Clear inputs
        clear_inputs()

    except ValueError as e:
        messagebox.showerror("Error", str(e))

# Function to clear input fields
def clear_inputs():
    name_entry.delete(0, tk.END)
    age_entry.delete(0, tk.END)
    profession_entry.delete(0, tk.END)
    gender_var.set("")
    gaming_experience_var.set("")

# Initialize the main application window
app = tk.Tk()
app.title("Participant Information Form")

# Create labels and entry fields for each participant's detail
tk.Label(app, text="Name:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
name_entry = tk.Entry(app)
name_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")

tk.Label(app, text="Age:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
age_entry = tk.Entry(app)
age_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")

tk.Label(app, text="Gender:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
gender_var = tk.StringVar()
tk.Radiobutton(app, text="Male", variable=gender_var, value="Male").grid(row=2, column=1, padx=10, pady=5, sticky="w")
tk.Radiobutton(app, text="Female", variable=gender_var, value="Female").grid(row=2, column=2, padx=10, pady=5, sticky="w")
tk.Radiobutton(app, text="Other", variable=gender_var, value="Other").grid(row=2, column=3, padx=10, pady=5, sticky="w")

tk.Label(app, text="Profession:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
profession_entry = tk.Entry(app)
profession_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")

tk.Label(app, text="Gaming Experience:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
gaming_experience_var = tk.StringVar()
tk.Radiobutton(app, text="Yes", variable=gaming_experience_var, value="Yes").grid(row=4, column=1, padx=10, pady=5, sticky="w")
tk.Radiobutton(app, text="No", variable=gaming_experience_var, value="No").grid(row=4, column=2, padx=10, pady=5, sticky="w")

# Button to submit the details
submit_button = tk.Button(app, text="Submit", command=save_to_csv)
submit_button.grid(row=5, column=0, columnspan=4, pady=10)

# Create CSV file with headers if it doesn't exist
csv_file = 'participants_info.csv'
try:
    with open(csv_file, mode='x', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Age", "Gender", "Profession", "Gaming Experience"])
except FileExistsError:
    pass

# Run the application
app.mainloop()
