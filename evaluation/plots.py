import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the .npy file
data = np.load("/home/viktoria/Git/KTH_DeepLearning-diffusion/evaluation/accuracy_val.npz.npy")


# Filenames of your .npy files
file_names = ['evaluation/accuracy.npz.npy', 'evaluation/accuracy_val.npz.npy', 'evaluation/loss.npz.npy', 'evaluation/loss_val.npz.npy']

# Titles for each plot
plot_titles = ['Train Accuracy', 'Validation Accuracy', 'Train Loss', 'Validation Loss']
plot_y_labels = ["Accuracy", "Accuracy", "Loss", "Loss"]

# Number of rows and columns for subplots
nrows = 2
ncols = 2

# Create a figure
plt.figure(figsize=(10, 8))

# Loop through each file
for i, file in enumerate(file_names):
    # Load the .npy file
    data = np.load(file)

    # Add a subplot
    plt.subplot(nrows, ncols, i + 1)

    # Plot the data
    # This example assumes 1D data, modify if your data is 2D or 3D
    plt.plot(data)

    # Add title
    plt.title(plot_titles[i])

    # Optionally set labels
    plt.xlabel('Epoch')
    plt.ylabel(plot_y_labels[i])

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()

