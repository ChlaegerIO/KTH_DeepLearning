import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define file groups for each subplot
# Each group contains file paths for the six models corresponding to one metric
file_groups = [
    ['evaluation/accuracy_ensemble_0.npz.npy', 'evaluation/accuracy_ensemble_1.npz.npy', 'evaluation/accuracy_ensemble_2.npz.npy', 'evaluation/accuracy_ensemble_3.npz.npy', 'evaluation/accuracy_ensemble_4.npz.npy', 'evaluation/accuracy_ensemble_5.npz.npy'],  # Train Accuracy for 6 models
    ['evaluation/accuracy_val_ensemble_0.npz.npy', 'evaluation/accuracy_val_ensemble_1.npz.npy', 'evaluation/accuracy_val_ensemble_2.npz.npy', 'evaluation/accuracy_val_ensemble_3.npz.npy', 'evaluation/accuracy_val_ensemble_4.npz.npy','evaluation/accuracy_val_ensemble_5.npz.npy'],      # Validation Accuracy for 6 models
    ['evaluation/loss_ensemble_0.npz.npy', 'evaluation/loss_ensemble_1.npz.npy', 'evaluation/loss_ensemble_2.npz.npy', 'evaluation/loss_ensemble_3.npz.npy', 'evaluation/loss_ensemble_4.npz.npy', 'evaluation/loss_ensemble_5.npz.npy'],         # Train Loss for 6 models
    ['evaluation/loss_val_ensemble_0.npz.npy', 'evaluation/loss_val_ensemble_1.npz.npy', 'evaluation/loss_val_ensemble_2.npz.npy', 'evaluation/loss_val_ensemble_3.npz.npy', 'evaluation/loss_val_ensemble_4.npz.npy', 'evaluation/loss_val_ensemble_5.npz.npy']              # Validation Loss for 6 models
]

# Titles and labels
plot_titles = ['Train Accuracy', 'Validation Accuracy', 'Train Loss', 'Validation Loss']
plot_y_labels = ["Accuracy", "Accuracy", "Loss", "Loss"]

# Subplot configuration
nrows, ncols = 2, 2

# Create a figure
plt.figure(figsize=(15, 10))

# Loop through each group of files (each metric)
for i, file_group in enumerate(file_groups):
    plt.subplot(nrows, ncols, i + 1)

    # Loop through each file (each model) in the group
    for j, file in enumerate(file_group):
        # Load the .npy file
        data = np.load(file)

        epochs = np.arange(1, len(data) + 1)
        plt.plot(epochs, data, label=f"Ensemble {j}")

    # Add title and labels
    plt.title(plot_titles[i])
    plt.xlabel('Epoch')
    plt.ylabel(plot_y_labels[i])
    plt.xscale('log')
    plt.legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()



# Filenames of your .npy files
file_names = ['evaluation/accuracy_ensemble_5.npz.npy', 'evaluation/accuracy_val_ensemble_5.npz.npy', 'evaluation/loss_ensemble_5.npz.npy', 'evaluation/loss_val_ensemble_5.npz.npy']

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

