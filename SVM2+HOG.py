import os
import sys
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import dump
# Function to load images from a directory
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
                        #img = img.convert('L')
                        img = img.resize((64, 64))
                        images.append(img)
    return images, labels


def extract_hog_features(images):
    features = []
    for img in images:
        img_gray = img.convert('L')  # Convert image to grayscale
        hog_features = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        features.append(hog_features)
        progress_bar.update(1)
    return features


# Specify the path to the folder containing your images
folder_path = "D:/DataSet/JiDaTop100/"

# Load images and labels from the specified folder
X, y = load_images_from_folder_assign_labels(folder_path)

#features = []
# for image in X:
#     img_gray = image.convert('L')
#
#     hog_features = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
#     features.append(hog_features)
#    # fd, hog_image = hog(img_gray, orientations=9, pixels_per_cell = (8,8), cells_per_block = (2,2), visualize = True, multichannel=True)
#    # hog_features.append(fd)
#
# hog_features = np.array(features)

# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(hog_features, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
progress_bar = tqdm(total=len(X_train), desc='Extracting HOG features from X_train', position=0, leave=True)
X_train_hog = extract_hog_features(X_train)
progress_bar.close()

progress_bar = tqdm(total=len(X_test), desc='Extracting HOG features from X_test', position=0, leave=True)
X_test_hog = extract_hog_features(X_test)
progress_bar.close()

# Create and train the SVM classifier
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train_hog, y_train)

# Make predictions on the test set
y_pred = svm_clf.predict(X_test_hog)

svm_model_size = sys.getsizeof(svm_clf)

print("Size of trained SVM model:", svm_model_size, "bytes")



svm_params = svm_clf.get_params()
print("SVM Model Parameters:")
for param, value in svm_params.items():
    print(f"{param}: {value}")
# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Visualize some of the images and their predicted labels
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i], cmap='gray')  # Modify the reshape size according to your image dimensions
    ax.set_title(f'Predicted: {y_pred[i]}')
    ax.axis('off')

plt.show()
dump(svm_clf, 'svm_Hog_model.joblib')