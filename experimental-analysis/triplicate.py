import csv

# Function to triplicate each row in a CSV file
def triplicate_csv(input_file, output_file):
    with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write the header
        header = next(reader)
        writer.writerow(header)
        
        # Triplicate each row
        for row in reader:
            for _ in range(3):  # Repeat each row 3 times
                writer.writerow(row)

# Example usage
input_file = 'participants_info.csv'  # Input CSV file
output_file = 'participants_info_triplicated.csv'  # Output CSV file with triplicated rows

triplicate_csv(input_file, output_file)
