import pandas as pd
import numpy as np

# Creating a sample test dataset
np.random.seed(42)  # Setting seed for reproducibility

# Generating random values for the sample test dataset
num_samples = 20
positions = np.random.uniform(0.5, 2.0, size=(num_samples, 2))  # Random (Position, Distance from Origin) pairs
labels = np.random.choice(['Heads', 'Tails'], size=num_samples)

# Creating a DataFrame
test_data = pd.DataFrame({
    'Drop ID': range(101, 101 + num_samples),
    'Image/Video File': [f'drop_{i:03d}.jpg' for i in range(101, 101 + num_samples)],
    'Initial Conditions': np.random.choice(['Head Facing Up', 'Vertical Drop', 'Tail Facing Up'], size=num_samples),
    'Coin Orientation': np.random.choice(['Head', 'Tail'], size=num_samples),
    'Position': [f'({pos[0]:.2f}, {pos[1]:.2f})' for pos in positions],
    'Distance from Origin': [f'{pos[1]:.2f} meters' for pos in positions],
    'Label (Heads/Tails)': labels
})

# Displaying the sample test dataset
print(test_data)
