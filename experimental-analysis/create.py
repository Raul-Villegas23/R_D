import pandas as pd
import random

# Original delays list
original_delays = [0, 0, 0, 0, 80, 80, 80, 80, 160, 160, 160, 160, 320, 320, 320, 320,
                   400, 400, 400, 400, 600, 600, 600, 600, 800, 800, 800, 800,
                   1000, 1000, 1000, 1000, 1200, 1200, 1200, 1200, 1400, 1400, 1400, 1400,
                   1600, 1600, 1600, 1600, 1800, 1800, 1800, 1800]

# Initialize data and shuffle delays
participants_data = []
remaining_delays = original_delays[:]

for i in range(1, 17):  # Participants 1 to 16
    # Shuffle remaining delays to ensure random selection
    random.shuffle(remaining_delays)

    # Select 3 unique delays for the current participant
    participant_delays = []
    while len(participant_delays) < 3:
        delay = random.choice(remaining_delays)
        if delay not in participant_delays:
            participant_delays.append(delay)

    # Ensure each delay is removed once used, maintaining overall counts
    for delay in participant_delays:
        remaining_delays.remove(delay)

    # Add the participant data
    for delay in participant_delays:
        participants_data.append([f'{i}', delay])

    # If we have depleted available unique combinations, reset the list
    if len(remaining_delays) < 3:
        remaining_delays = original_delays[:]

# Create DataFrame
df = pd.DataFrame(participants_data, columns=['Participants', 'Delays'])

# Save to CSV
df.to_csv('participants_delays.csv', index=False)

print("CSV file has been created with unique delays for each participant.")
