import random

# Generate 48 random times between 1 minute (60 seconds) and 2 minutes (120 seconds)
times = []
for _ in range(48):
    # Random second between 60 and 120
    total_seconds = random.randint(60, 120)
    # Convert to minutes and seconds
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    # Random milliseconds from 0 to 99
    milliseconds = random.randint(0, 99)

    # Format time as mm:ss:sss
    formatted_time = f'{minutes:02}:{seconds:02}:{milliseconds:03}'
    times.append(formatted_time)

# Print the list of times
for time in times:
    print(time)
