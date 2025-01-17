import matplotlib.pyplot as plt
import numpy as np
import json

import os


window_size= 20
def moving_average(data, window_size):
    """Apply a moving average to the data."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')



plt.figure(figsize=(8, 6))

# Specify the directory path
directory_path = "exam/src/A/data"

# Loop through all files in the directory

# Loop through all files in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    # Check if it's a file (not a directory)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            dct = json.load(file)

        model = dct["model"]
        epochs = []
        loss = []
        for i in range(1, 101):
            si = str(i)
            epochs.append(dct[si]["epoch"]+window_size-1)
            loss.append(dct[si]["FID"])

        print(epochs[:5], loss[:5])  # Sanity check for data

        # Apply smoothing (choose one)
        smoothed_loss = moving_average(loss, window_size=window_size)  # Moving average
        # smoothed_loss = savgol_filter(loss, window_length=11, polyorder=2)  # Savitzky-Golay

        # Adjust epochs for moving average (due to reduced length)
        adjusted_epochs = epochs[:len(smoothed_loss)]

        # Plot the smoothed data
        plt.plot(adjusted_epochs, smoothed_loss, label=model)
    
# Formatting
#plt.yscale('log')
plt.xlabel('Training Epochs', fontsize=12)
plt.ylabel('FID Estimate', fontsize=12)
plt.title('FID curves', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()





plt.figure(figsize=(8, 6))
# Loop through all files in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    # Check if it's a file (not a directory)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            dct = json.load(file)

        model = dct["model"]
        epochs = []
        loss = []
        for i in range(1, 101):
            si = str(i)
            epochs.append(dct[si]["epoch"]+window_size-1)
            loss.append(dct[si]["Inception Score"])

        print(epochs[:5], loss[:5])  # Sanity check for data

        # Apply smoothing (choose one)
        smoothed_loss = moving_average(loss, window_size=window_size)  # Moving average
        # smoothed_loss = savgol_filter(loss, window_length=11, polyorder=2)  # Savitzky-Golay

        # Adjust epochs for moving average (due to reduced length)
        adjusted_epochs = epochs[:len(smoothed_loss)]

        # Plot the smoothed data
        plt.plot(adjusted_epochs, smoothed_loss, label=model)
    
# Formatting
#plt.yscale('log')
plt.xlabel('Training Epochs', fontsize=12)
plt.ylabel('Inception Score Estimate', fontsize=12)
plt.title(' Inception Score curves', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()






# Show the plot
plt.show()