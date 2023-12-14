import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


PLOT_IMAGE_GRID = False
PLOT_EVAL = False

print(np.load("evaluation/paperWithDG/eval_metrics.npy"))

# plot 8x8 images
if PLOT_IMAGE_GRID:
    images = []
    for i in range(64):
        images.append(np.load(f'data/generated_samples_cifar10_paperWithDG/0000{i}.npz')["samples"])
        images = np.array(images)
    # images = np.load("data/generated_samples_cifar10_paperWithDG/000064.npz")["samples"]              # for batched immages

    margin=1 # pixels
    spacing = 1/200 # pixels
    dpi=100. # dots per inch

    width = (400+180+2*margin+spacing)/dpi # inches
    height= (400+180+2*margin+spacing)/dpi

    left = spacing #axes ratio
    bottom = spacing
    wspace = spacing

    fig, axes  = plt.subplots(8,8, figsize=(width,height), dpi=dpi)
    fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom, 
                        wspace=wspace, hspace=wspace)

    for ax, im in zip(axes.flatten(),images):
        ax.axis('off')
        ax.imshow(im)

    plt.show()

    # save plot
    fig.savefig('evaluation/OurEnsemble/sample_images.png', dpi=dpi, bbox_inches='tight')


# Define file groups for each subplot
# Each group contains file paths for the six models corresponding to one metric
if PLOT_EVAL:
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
