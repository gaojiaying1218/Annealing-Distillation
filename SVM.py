import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os
from PIL import Image
import seaborn as sns

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Function to convert images to numpy arrays
def load_images_from_folder_assign_labels(root_dir):
    """
    Assign labels based on subfolder names in the root directory.

    Args:
    root_dir (str): The root directory path.

    Returns:
    dict: A dictionary mapping file paths to their corresponding labels.
    """
    labels = []
    images = []
    for label, subfolder in enumerate(sorted(os.listdir(root_dir))):  # Enumerate over subfolders
        subfolder_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                if os.path.isfile(file_path):  # Check if it's a file
                    #labels[file_path] = label
                    labels.append(label)
                img = Image.open(file_path)
                if img is not None:
                    img = img.convert('L')
                    img = img.resize((128, 128))
                    images.append(img)
    return images,labels



def images_to_arrays(images):
    arrays = []
    for img in images:
        array = np.array(img)
        arrays.append(array.flatten())
    return arrays

# Load images from a folder
folder_path = "D:/DataSet/JiDaTop5/"
images, label = load_images_from_folder_assign_labels(folder_path)

# Convert images to numpy arrays
image_arrays = images_to_arrays(images)
y = np.array(label)
# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(image_arrays)

# Reduce dimensionality using PCA
pca = PCA(n_components=50)  # Adjust the number of components as needed
reduced_data = pca.fit_transform(scaled_data)

# Define and train SVM
svm = SVC(kernel='rbf', gamma='scale')  # You can adjust parameters like kernel and gamma
svm.fit(reduced_data,y)

# Get cluster labels from SVM decision function
labels = svm.decision_function(reduced_data)


# Visualize the clustering results
for label in np.unique(labels):
    cluster_images = [images[i] for i in range(len(images)) if labels[i].all() == label]
    fig, axes = plt.subplots(1, len(cluster_images), figsize=(10, 5))
    for i, img in enumerate(cluster_images):
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.show()
