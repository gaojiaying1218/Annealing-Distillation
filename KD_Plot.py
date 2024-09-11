import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms,datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
import joblib

class TeacherModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TeacherModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class StudentModel(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4096, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, num_classes)

    def forward(self, x):
        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs):
    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(data)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Calculate average loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation loop
        model.eval()
        num_correct = 0
        num_samples = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                predictions = preds.max(1).indices
                num_correct += (predictions == y).sum().item()
                num_samples += predictions.size(0)

        accuracy = num_correct / num_samples
        test_accuracies.append(accuracy)
        print(f'Model: {model.__class__.__name__}\t Epoch: {epoch + 1}\t Loss: {avg_train_loss:.4f}\t Accuracy: {accuracy:.4f}')


    return train_losses, test_accuracies, model

def KD_train_model(model, teacher_model, train_loader, test_loader, criterion, optimizer, device, epochs):
    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Teacher model predictions
            with torch.no_grad():
                teacher_preds = teacher_model(data)

            # Student model predictions
            student_preds = model(data)

            # Calculate hard loss
            student_loss = criterion(student_preds, targets)

            # Calculate soft loss for knowledge distillation
            distillation_loss = kd_loss(
                F.log_softmax(student_preds / temp, dim=1),
                F.softmax(teacher_preds / temp, dim=1) * temp ** 2
            )

            # Weighted sum of hard and soft losses
            loss = alpha * student_loss + (1 - alpha) * distillation_loss

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Average loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation on test set
        model.eval()
        num_correct = 0
        num_samples = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                preds = model(x)
                predictions = preds.max(1).indices
                num_correct += (predictions == y).sum().item()
                num_samples += predictions.size(0)

        accuracy = num_correct / num_samples
        test_accuracies.append(accuracy)

        #model.train()
        print(f'Model: {model.__class__.__name__}\t Epoch: {epoch + 1}\t Loss: {avg_train_loss:.4f}\t Accuracy: {accuracy:.4f}')



    return train_losses, test_accuracies

# Function to calculate distances to cluster centers
def calculate_distances_to_centers(vector, centers):
    distances = np.linalg.norm(centers - vector, axis=1)
    return distances


def KD_from_kMeans_train_model(model, kmeans, value, train_loader, test_loader, optimizer, device, epochs):
    train_losses = []
    test_accuracies = []
    # Define loss functions and optimizer
    temp = 3
    alpha_k = value
    hard_loss = nn.CrossEntropyLoss()
    kd_loss = nn.KLDivLoss(reduction='batchmean')
    #optimizer = Adam(student_model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for data, targets in tqdm(train_loader):
            data, targets = data.to(device), targets.to(device)

            hog_features_batch = []
            soft_labels_batch = []

            for img in data:
                img_np = img.squeeze().cpu().numpy()  # Convert tensor to numpy array
                hog_features, _ = hog(img_np, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
                hog_features_batch.append(hog_features)

                distances = calculate_distances_to_centers(hog_features, kmeans.cluster_centers_)
                soft_labels_batch.append(distances)
                soft_labels_batch_np = np.array(soft_labels_batch)


            soft_labels_batch = torch.tensor(soft_labels_batch_np, dtype=torch.float32).to(device)

            optimizer.zero_grad()

            student_preds = model(data)

            student_loss = hard_loss(student_preds, targets)
            distillation_loss = kd_loss(
                F.log_softmax(student_preds / temp, dim=1),
                F.softmax(soft_labels_batch / temp, dim=1)
            ) * (temp ** 2)
            #
            # if epoch < epochs/3:
            #     alpha_k = 0.75
            # if epochs/3 <= epoch < epochs*2/3:
            #     alpha_k = 0.75
            # if epochs*2/3 <= epoch :
            #     alpha_k = 0.75



            loss = alpha_k * student_loss + (1 - alpha_k) * distillation_loss
            # print(student_loss.item())
            # print(distillation_loss.item())
            # print(loss.item())

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Average loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate the model
        model.eval()
        num_correct = 0
        num_samples = 0

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)

                preds = model(data)
                predictions = preds.max(1).indices
                num_correct += (predictions == targets).sum().item()
                num_samples += predictions.size(0)
            acc = (num_correct / num_samples)

           #num_correct += (predictions == y).sum().item()
        #num_samples += predictions.size(0)

        #acc = num_correct / num_samples
            test_accuracies.append(acc)

        # model.train()
        print(f'Model: Kmeans \t Epoch: {epoch + 1}\t Loss: {avg_train_loss:.4f}\t Accuracy: {acc:.4f}')


    return train_losses, test_accuracies

# Initialize models, dataloaders, criterion, optimizer, etc.
##
##
##



# Define hyperparameters
temp = 3
alpha = 0.1
kd_loss = nn.KLDivLoss(reduction="batchmean")
criterion = nn.CrossEntropyLoss()
# optimizer_original = optim.Adam(original_model.parameters(), lr=1e-4)
# optimizer_student_B = optim.Adam(student_model_B.parameters(), lr=1e-4)



# Load your local dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),# Resize images to fit ResNet input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#dataset = ImageFolder(root='D:/DataSet/JiDaTop10/', transform=transform)
# Load the MNIST dataset
#dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# Load the CIFAR-10 dataset
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


# Split dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



epochs = 30
colors = ['r', 'g', 'b', 'k']


plt.figure(figsize=(12, 5))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for k in range(11):

    # Initialize models, dataloaders, criterion, optimizer, etc.

    TeacherModelCNN = TeacherModel().to(device)
    StudentModelA = StudentModel().to(device)
    StudentModelB = StudentModel().to(device)
    StudentModelC = StudentModel().to(device)

    models = [TeacherModelCNN, StudentModelA, StudentModelB, StudentModelC]
    criterion = nn.CrossEntropyLoss()
    optimizers = [optim.Adam(model.parameters(), lr=1e-4) for model in models]

    value = 1-(k/10.0)
    for i, model in enumerate(models):
        model_name = [name for name, value in locals().items() if value is model][0]
        print(f"Model {i + 1} name:", model_name)

        if model ==TeacherModelCNN:
            train_losses, test_accuracies, teacher = train_model(model, train_loader, test_loader, criterion, optimizers[i], device, epochs)
            max_train_loss = max(train_losses)
            normalized_train_losses = [loss / max_train_loss for loss in train_losses]

        if model == StudentModelA:
            train_losses, test_accuracies, _ = train_model(model, train_loader, test_loader, criterion, optimizers[i], device, epochs)
            max_train_loss = max(train_losses)
            normalized_train_losses = [loss / max_train_loss for loss in train_losses]

        if model == StudentModelB:
            train_losses, test_accuracies = KD_train_model(model, teacher, train_loader, test_loader, criterion, optimizers[i], device, epochs)
            max_train_loss = max(train_losses)
            normalized_train_losses = [loss / max_train_loss for loss in train_losses]

        if model == StudentModelC:
            kmeans = joblib.load('K-model_cifar10.pkl1')
            train_losses, test_accuracies = KD_from_kMeans_train_model(model, kmeans, value, train_loader, test_loader, optimizers[i], device, epochs)
            max_train_loss = max(train_losses)
            normalized_train_losses = [loss / max_train_loss for loss in train_losses]

        plt.subplot(1, 2, 1)
        plt.plot(normalized_train_losses, label=f'{model_name} Training Loss', color=colors[i])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(test_accuracies, label=f'{model_name} Test Accuracy', color=colors[i])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Test Accuracy Curve')
        plt.legend()

    filename = f"./cifar10_alpha_value/alpha_value_{value}.png"
    plt.savefig(filename)
    plt.clf()
    print(f"Save plot as {filename}")
    #plt.show()

# D:\APP\Anaconda\envs\d2l\python.exe D:\Code\HOG_SVM\KD_Plot.py
# Model 1 name: TeacherModelCNN
# Model: TeacherModel	 Epoch: 1	 Loss: 0.4892	 Accuracy: 0.9299
# Model: TeacherModel	 Epoch: 2	 Loss: 0.2569	 Accuracy: 0.9432
# Model: TeacherModel	 Epoch: 3	 Loss: 0.2050	 Accuracy: 0.9558
# Model: TeacherModel	 Epoch: 4	 Loss: 0.1787	 Accuracy: 0.9598
# Model: TeacherModel	 Epoch: 5	 Loss: 0.1617	 Accuracy: 0.9637
# Model: TeacherModel	 Epoch: 6	 Loss: 0.1474	 Accuracy: 0.9660
# Model: TeacherModel	 Epoch: 7	 Loss: 0.1389	 Accuracy: 0.9702
# Model: TeacherModel	 Epoch: 8	 Loss: 0.1266	 Accuracy: 0.9680
# Model: TeacherModel	 Epoch: 9	 Loss: 0.1243	 Accuracy: 0.9700
# Model: TeacherModel	 Epoch: 10	 Loss: 0.1135	 Accuracy: 0.9708
# Model: TeacherModel	 Epoch: 11	 Loss: 0.1085	 Accuracy: 0.9728
# Model: TeacherModel	 Epoch: 12	 Loss: 0.1045	 Accuracy: 0.9723
# Model: TeacherModel	 Epoch: 13	 Loss: 0.0990	 Accuracy: 0.9743
# Model: TeacherModel	 Epoch: 14	 Loss: 0.0939	 Accuracy: 0.9730
# Model: TeacherModel	 Epoch: 15	 Loss: 0.0929	 Accuracy: 0.9740
# Model: TeacherModel	 Epoch: 16	 Loss: 0.0900	 Accuracy: 0.9762
# Model: TeacherModel	 Epoch: 17	 Loss: 0.0876	 Accuracy: 0.9776
# Model: TeacherModel	 Epoch: 18	 Loss: 0.0837	 Accuracy: 0.9784
# Model: TeacherModel	 Epoch: 19	 Loss: 0.0791	 Accuracy: 0.9773
# Model: TeacherModel	 Epoch: 20	 Loss: 0.0760	 Accuracy: 0.9774
# Model: TeacherModel	 Epoch: 21	 Loss: 0.0801	 Accuracy: 0.9770
# Model: TeacherModel	 Epoch: 22	 Loss: 0.0722	 Accuracy: 0.9777
# Model: TeacherModel	 Epoch: 23	 Loss: 0.0717	 Accuracy: 0.9783
# Model: TeacherModel	 Epoch: 24	 Loss: 0.0733	 Accuracy: 0.9801
# Model: TeacherModel	 Epoch: 25	 Loss: 0.0685	 Accuracy: 0.9772
# Model: TeacherModel	 Epoch: 26	 Loss: 0.0671	 Accuracy: 0.9785
# Model: TeacherModel	 Epoch: 27	 Loss: 0.0683	 Accuracy: 0.9792
# Model: TeacherModel	 Epoch: 28	 Loss: 0.0665	 Accuracy: 0.9796
# Model: TeacherModel	 Epoch: 29	 Loss: 0.0651	 Accuracy: 0.9798
# Model: TeacherModel	 Epoch: 30	 Loss: 0.0632	 Accuracy: 0.9781
# Model: TeacherModel	 Epoch: 31	 Loss: 0.0640	 Accuracy: 0.9792
# Model: TeacherModel	 Epoch: 32	 Loss: 0.0607	 Accuracy: 0.9806
# Model: TeacherModel	 Epoch: 33	 Loss: 0.0604	 Accuracy: 0.9821
# Model: TeacherModel	 Epoch: 34	 Loss: 0.0572	 Accuracy: 0.9798
# Model: TeacherModel	 Epoch: 35	 Loss: 0.0567	 Accuracy: 0.9806
# Model: TeacherModel	 Epoch: 36	 Loss: 0.0570	 Accuracy: 0.9802
# Model: TeacherModel	 Epoch: 37	 Loss: 0.0575	 Accuracy: 0.9795
# Model: TeacherModel	 Epoch: 38	 Loss: 0.0556	 Accuracy: 0.9814
# Model: TeacherModel	 Epoch: 39	 Loss: 0.0536	 Accuracy: 0.9807
# Model: TeacherModel	 Epoch: 40	 Loss: 0.0534	 Accuracy: 0.9813
# Model: TeacherModel	 Epoch: 41	 Loss: 0.0532	 Accuracy: 0.9822
# Model: TeacherModel	 Epoch: 42	 Loss: 0.0524	 Accuracy: 0.9812
# Model: TeacherModel	 Epoch: 43	 Loss: 0.0525	 Accuracy: 0.9818
# Model: TeacherModel	 Epoch: 44	 Loss: 0.0481	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 45	 Loss: 0.0510	 Accuracy: 0.9817
# Model: TeacherModel	 Epoch: 46	 Loss: 0.0480	 Accuracy: 0.9819
# Model: TeacherModel	 Epoch: 47	 Loss: 0.0501	 Accuracy: 0.9803
# Model: TeacherModel	 Epoch: 48	 Loss: 0.0498	 Accuracy: 0.9811
# Model: TeacherModel	 Epoch: 49	 Loss: 0.0504	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 50	 Loss: 0.0467	 Accuracy: 0.9783
# Model 2 name: StudentModelA
# Model: StudentModel	 Epoch: 1	 Loss: 0.9922	 Accuracy: 0.8488
# Model: StudentModel	 Epoch: 2	 Loss: 0.4453	 Accuracy: 0.8838
# Model: StudentModel	 Epoch: 3	 Loss: 0.3780	 Accuracy: 0.8938
# Model: StudentModel	 Epoch: 4	 Loss: 0.3488	 Accuracy: 0.8982
# Model: StudentModel	 Epoch: 5	 Loss: 0.3292	 Accuracy: 0.9019
# Model: StudentModel	 Epoch: 6	 Loss: 0.3145	 Accuracy: 0.9062
# Model: StudentModel	 Epoch: 7	 Loss: 0.3034	 Accuracy: 0.9093
# Model: StudentModel	 Epoch: 8	 Loss: 0.2928	 Accuracy: 0.9115
# Model: StudentModel	 Epoch: 9	 Loss: 0.2846	 Accuracy: 0.9141
# Model: StudentModel	 Epoch: 10	 Loss: 0.2763	 Accuracy: 0.9153
# Model: StudentModel	 Epoch: 11	 Loss: 0.2701	 Accuracy: 0.9161
# Model: StudentModel	 Epoch: 12	 Loss: 0.2630	 Accuracy: 0.9184
# Model: StudentModel	 Epoch: 13	 Loss: 0.2575	 Accuracy: 0.9198
# Model: StudentModel	 Epoch: 14	 Loss: 0.2516	 Accuracy: 0.9202
# Model: StudentModel	 Epoch: 15	 Loss: 0.2469	 Accuracy: 0.9237
# Model: StudentModel	 Epoch: 16	 Loss: 0.2408	 Accuracy: 0.9257
# Model: StudentModel	 Epoch: 17	 Loss: 0.2374	 Accuracy: 0.9263
# Model: StudentModel	 Epoch: 18	 Loss: 0.2323	 Accuracy: 0.9234
# Model: StudentModel	 Epoch: 19	 Loss: 0.2284	 Accuracy: 0.9215
# Model: StudentModel	 Epoch: 20	 Loss: 0.2249	 Accuracy: 0.9283
# Model: StudentModel	 Epoch: 21	 Loss: 0.2218	 Accuracy: 0.9277
# Model: StudentModel	 Epoch: 22	 Loss: 0.2183	 Accuracy: 0.9291
# Model: StudentModel	 Epoch: 23	 Loss: 0.2160	 Accuracy: 0.9267
# Model: StudentModel	 Epoch: 24	 Loss: 0.2123	 Accuracy: 0.9286
# Model: StudentModel	 Epoch: 25	 Loss: 0.2097	 Accuracy: 0.9303
# Model: StudentModel	 Epoch: 26	 Loss: 0.2073	 Accuracy: 0.9327
# Model: StudentModel	 Epoch: 27	 Loss: 0.2048	 Accuracy: 0.9332
# Model: StudentModel	 Epoch: 28	 Loss: 0.2026	 Accuracy: 0.9289
# Model: StudentModel	 Epoch: 29	 Loss: 0.2004	 Accuracy: 0.9333
# Model: StudentModel	 Epoch: 30	 Loss: 0.1985	 Accuracy: 0.9292
# Model: StudentModel	 Epoch: 31	 Loss: 0.1964	 Accuracy: 0.9336
# Model: StudentModel	 Epoch: 32	 Loss: 0.1948	 Accuracy: 0.9323
# Model: StudentModel	 Epoch: 33	 Loss: 0.1932	 Accuracy: 0.9323
# Model: StudentModel	 Epoch: 34	 Loss: 0.1914	 Accuracy: 0.9332
# Model: StudentModel	 Epoch: 35	 Loss: 0.1895	 Accuracy: 0.9333
# Model: StudentModel	 Epoch: 36	 Loss: 0.1878	 Accuracy: 0.9356
# Model: StudentModel	 Epoch: 37	 Loss: 0.1873	 Accuracy: 0.9356
# Model: StudentModel	 Epoch: 38	 Loss: 0.1853	 Accuracy: 0.9372
# Model: StudentModel	 Epoch: 39	 Loss: 0.1839	 Accuracy: 0.9360
# Model: StudentModel	 Epoch: 40	 Loss: 0.1830	 Accuracy: 0.9347
# Model: StudentModel	 Epoch: 41	 Loss: 0.1812	 Accuracy: 0.9349
# Model: StudentModel	 Epoch: 42	 Loss: 0.1799	 Accuracy: 0.9388
# Model: StudentModel	 Epoch: 43	 Loss: 0.1799	 Accuracy: 0.9376
# Model: StudentModel	 Epoch: 44	 Loss: 0.1786	 Accuracy: 0.9327
# Model: StudentModel	 Epoch: 45	 Loss: 0.1769	 Accuracy: 0.9390
# Model: StudentModel	 Epoch: 46	 Loss: 0.1754	 Accuracy: 0.9381
# Model: StudentModel	 Epoch: 47	 Loss: 0.1754	 Accuracy: 0.9394
# Model: StudentModel	 Epoch: 48	 Loss: 0.1738	 Accuracy: 0.9390
# Model: StudentModel	 Epoch: 49	 Loss: 0.1731	 Accuracy: 0.9373
# Model: StudentModel	 Epoch: 50	 Loss: 0.1712	 Accuracy: 0.9387
# Model 3 name: StudentModelB
# Model: StudentModel	 Epoch: 1	 Loss: 25.4330	 Accuracy: 0.8436
# Model: StudentModel	 Epoch: 2	 Loss: 20.7518	 Accuracy: 0.8761
# Model: StudentModel	 Epoch: 3	 Loss: 20.2201	 Accuracy: 0.8895
# Model: StudentModel	 Epoch: 4	 Loss: 20.0040	 Accuracy: 0.8952
# Model: StudentModel	 Epoch: 5	 Loss: 19.8737	 Accuracy: 0.8976
# Model: StudentModel	 Epoch: 6	 Loss: 19.7688	 Accuracy: 0.9023
# Model: StudentModel	 Epoch: 7	 Loss: 19.6865	 Accuracy: 0.9077
# Model: StudentModel	 Epoch: 8	 Loss: 19.6169	 Accuracy: 0.9065
# Model: StudentModel	 Epoch: 9	 Loss: 19.5609	 Accuracy: 0.9067
# Model: StudentModel	 Epoch: 10	 Loss: 19.5124	 Accuracy: 0.9120
# Model: StudentModel	 Epoch: 11	 Loss: 19.4716	 Accuracy: 0.9140
# Model: StudentModel	 Epoch: 12	 Loss: 19.4286	 Accuracy: 0.9155
# Model: StudentModel	 Epoch: 13	 Loss: 19.3898	 Accuracy: 0.9157
# Model: StudentModel	 Epoch: 14	 Loss: 19.3580	 Accuracy: 0.9169
# Model: StudentModel	 Epoch: 15	 Loss: 19.3263	 Accuracy: 0.9183
# Model: StudentModel	 Epoch: 16	 Loss: 19.2970	 Accuracy: 0.9188
# Model: StudentModel	 Epoch: 17	 Loss: 19.2651	 Accuracy: 0.9203
# Model: StudentModel	 Epoch: 18	 Loss: 19.2358	 Accuracy: 0.9217
# Model: StudentModel	 Epoch: 19	 Loss: 19.2056	 Accuracy: 0.9216
# Model: StudentModel	 Epoch: 20	 Loss: 19.1824	 Accuracy: 0.9223
# Model: StudentModel	 Epoch: 21	 Loss: 19.1601	 Accuracy: 0.9256
# Model: StudentModel	 Epoch: 22	 Loss: 19.1400	 Accuracy: 0.9249
# Model: StudentModel	 Epoch: 23	 Loss: 19.1161	 Accuracy: 0.9247
# Model: StudentModel	 Epoch: 24	 Loss: 19.1011	 Accuracy: 0.9282
# Model: StudentModel	 Epoch: 25	 Loss: 19.0808	 Accuracy: 0.9254
# Model: StudentModel	 Epoch: 26	 Loss: 19.0651	 Accuracy: 0.9263
# Model: StudentModel	 Epoch: 27	 Loss: 19.0489	 Accuracy: 0.9287
# Model: StudentModel	 Epoch: 28	 Loss: 19.0332	 Accuracy: 0.9297
# Model: StudentModel	 Epoch: 29	 Loss: 19.0223	 Accuracy: 0.9310
# Model: StudentModel	 Epoch: 30	 Loss: 19.0063	 Accuracy: 0.9285
# Model: StudentModel	 Epoch: 31	 Loss: 18.9947	 Accuracy: 0.9315
# Model: StudentModel	 Epoch: 32	 Loss: 18.9853	 Accuracy: 0.9320
# Model: StudentModel	 Epoch: 33	 Loss: 18.9763	 Accuracy: 0.9323
# Model: StudentModel	 Epoch: 34	 Loss: 18.9640	 Accuracy: 0.9323
# Model: StudentModel	 Epoch: 35	 Loss: 18.9510	 Accuracy: 0.9327
# Model: StudentModel	 Epoch: 36	 Loss: 18.9448	 Accuracy: 0.9307
# Model: StudentModel	 Epoch: 37	 Loss: 18.9339	 Accuracy: 0.9321
# Model: StudentModel	 Epoch: 38	 Loss: 18.9228	 Accuracy: 0.9321
# Model: StudentModel	 Epoch: 39	 Loss: 18.9164	 Accuracy: 0.9341
# Model: StudentModel	 Epoch: 40	 Loss: 18.9060	 Accuracy: 0.9339
# Model: StudentModel	 Epoch: 41	 Loss: 18.8990	 Accuracy: 0.9337
# Model: StudentModel	 Epoch: 42	 Loss: 18.8915	 Accuracy: 0.9344
# Model: StudentModel	 Epoch: 43	 Loss: 18.8854	 Accuracy: 0.9355
# Model: StudentModel	 Epoch: 44	 Loss: 18.8753	 Accuracy: 0.9283
# Model: StudentModel	 Epoch: 45	 Loss: 18.8691	 Accuracy: 0.9362
# Model: StudentModel	 Epoch: 46	 Loss: 18.8605	 Accuracy: 0.9363
# Model: StudentModel	 Epoch: 47	 Loss: 18.8551	 Accuracy: 0.9347
# Model: StudentModel	 Epoch: 48	 Loss: 18.8465	 Accuracy: 0.9343
# Model: StudentModel	 Epoch: 49	 Loss: 18.8393	 Accuracy: 0.9359
# Model: StudentModel	 Epoch: 50	 Loss: 18.8329	 Accuracy: 0.9344
# Model 4 name: StudentModelC
# 100%|██████████| 1500/1500 [14:22<00:00,  1.74it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 1	 Loss: 0.8939	 Accuracy: 0.8724
# 100%|██████████| 1500/1500 [09:37<00:00,  2.60it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 2	 Loss: 0.4100	 Accuracy: 0.8942
# 100%|██████████| 1500/1500 [09:18<00:00,  2.68it/s]
# Model: Kmeans 	 Epoch: 3	 Loss: 0.3528	 Accuracy: 0.8999
# 100%|██████████| 1500/1500 [09:33<00:00,  2.61it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 4	 Loss: 0.3274	 Accuracy: 0.9032
# 100%|██████████| 1500/1500 [09:20<00:00,  2.67it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 5	 Loss: 0.3108	 Accuracy: 0.9079
# 100%|██████████| 1500/1500 [09:30<00:00,  2.63it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 6	 Loss: 0.2980	 Accuracy: 0.9107
# 100%|██████████| 1500/1500 [09:28<00:00,  2.64it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 7	 Loss: 0.2865	 Accuracy: 0.9147
# 100%|██████████| 1500/1500 [13:37<00:00,  1.84it/s]
# Model: Kmeans 	 Epoch: 8	 Loss: 0.2780	 Accuracy: 0.9167
# 100%|██████████| 1500/1500 [15:03<00:00,  1.66it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 9	 Loss: 0.2690	 Accuracy: 0.9179
# 100%|██████████| 1500/1500 [14:33<00:00,  1.72it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 10	 Loss: 0.2614	 Accuracy: 0.9201
# 100%|██████████| 1500/1500 [14:50<00:00,  1.68it/s]
# Model: Kmeans 	 Epoch: 11	 Loss: 0.2551	 Accuracy: 0.9220
# 100%|██████████| 1500/1500 [13:03<00:00,  1.92it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 12	 Loss: 0.2485	 Accuracy: 0.9201
# 100%|██████████| 1500/1500 [13:13<00:00,  1.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 13	 Loss: 0.2426	 Accuracy: 0.9224
# 100%|██████████| 1500/1500 [15:13<00:00,  1.64it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 14	 Loss: 0.2382	 Accuracy: 0.9228
# 100%|██████████| 1500/1500 [12:17<00:00,  2.03it/s]
# Model: Kmeans 	 Epoch: 15	 Loss: 0.2327	 Accuracy: 0.9253
# 100%|██████████| 1500/1500 [09:25<00:00,  2.65it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 16	 Loss: 0.2281	 Accuracy: 0.9256
# 100%|██████████| 1500/1500 [09:27<00:00,  2.64it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 17	 Loss: 0.2239	 Accuracy: 0.9253
# 100%|██████████| 1500/1500 [09:20<00:00,  2.68it/s]
# Model: Kmeans 	 Epoch: 18	 Loss: 0.2198	 Accuracy: 0.9268
# 100%|██████████| 1500/1500 [09:28<00:00,  2.64it/s]
# Model: Kmeans 	 Epoch: 19	 Loss: 0.2158	 Accuracy: 0.9285
# 100%|██████████| 1500/1500 [09:53<00:00,  2.53it/s]
# Model: Kmeans 	 Epoch: 20	 Loss: 0.2124	 Accuracy: 0.9301
# 100%|██████████| 1500/1500 [12:36<00:00,  1.98it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 21	 Loss: 0.2095	 Accuracy: 0.9295
# 100%|██████████| 1500/1500 [12:48<00:00,  1.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 22	 Loss: 0.2060	 Accuracy: 0.9278
# 100%|██████████| 1500/1500 [12:49<00:00,  1.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 23	 Loss: 0.2032	 Accuracy: 0.9307
# 100%|██████████| 1500/1500 [12:57<00:00,  1.93it/s]
# Model: Kmeans 	 Epoch: 24	 Loss: 0.2005	 Accuracy: 0.9315
# 100%|██████████| 1500/1500 [12:55<00:00,  1.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 25	 Loss: 0.1980	 Accuracy: 0.9339
# 100%|██████████| 1500/1500 [13:12<00:00,  1.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 26	 Loss: 0.1950	 Accuracy: 0.9347
# 100%|██████████| 1500/1500 [13:02<00:00,  1.92it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 27	 Loss: 0.1921	 Accuracy: 0.9337
# 100%|██████████| 1500/1500 [12:47<00:00,  1.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 28	 Loss: 0.1905	 Accuracy: 0.9355
# 100%|██████████| 1500/1500 [12:58<00:00,  1.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 29	 Loss: 0.1884	 Accuracy: 0.9354
# 100%|██████████| 1500/1500 [12:37<00:00,  1.98it/s]
# Model: Kmeans 	 Epoch: 30	 Loss: 0.1859	 Accuracy: 0.9367
# 100%|██████████| 1500/1500 [12:55<00:00,  1.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 31	 Loss: 0.1838	 Accuracy: 0.9342
# 100%|██████████| 1500/1500 [12:50<00:00,  1.95it/s]
# Model: Kmeans 	 Epoch: 32	 Loss: 0.1814	 Accuracy: 0.9391
# 100%|██████████| 1500/1500 [13:18<00:00,  1.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 33	 Loss: 0.1788	 Accuracy: 0.9343
# 100%|██████████| 1500/1500 [12:58<00:00,  1.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 34	 Loss: 0.1778	 Accuracy: 0.9388
# 100%|██████████| 1500/1500 [12:55<00:00,  1.94it/s]
# Model: Kmeans 	 Epoch: 35	 Loss: 0.1760	 Accuracy: 0.9372
# 100%|██████████| 1500/1500 [13:11<00:00,  1.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 36	 Loss: 0.1746	 Accuracy: 0.9357
# 100%|██████████| 1500/1500 [12:55<00:00,  1.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 37	 Loss: 0.1725	 Accuracy: 0.9374
# 100%|██████████| 1500/1500 [12:31<00:00,  2.00it/s]
# Model: Kmeans 	 Epoch: 38	 Loss: 0.1714	 Accuracy: 0.9358
# 100%|██████████| 1500/1500 [12:34<00:00,  1.99it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 39	 Loss: 0.1699	 Accuracy: 0.9373
# 100%|██████████| 1500/1500 [12:56<00:00,  1.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 40	 Loss: 0.1681	 Accuracy: 0.9406
# 100%|██████████| 1500/1500 [12:40<00:00,  1.97it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 41	 Loss: 0.1667	 Accuracy: 0.9414
# 100%|██████████| 1500/1500 [12:51<00:00,  1.94it/s]
# Model: Kmeans 	 Epoch: 42	 Loss: 0.1650	 Accuracy: 0.9361
# 100%|██████████| 1500/1500 [12:42<00:00,  1.97it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 43	 Loss: 0.1639	 Accuracy: 0.9387
# 100%|██████████| 1500/1500 [12:52<00:00,  1.94it/s]
# Model: Kmeans 	 Epoch: 44	 Loss: 0.1639	 Accuracy: 0.9420
# 100%|██████████| 1500/1500 [13:05<00:00,  1.91it/s]
# Model: Kmeans 	 Epoch: 45	 Loss: 0.1615	 Accuracy: 0.9429
# 100%|██████████| 1500/1500 [13:12<00:00,  1.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 46	 Loss: 0.1609	 Accuracy: 0.9429
# 100%|██████████| 1500/1500 [12:53<00:00,  1.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 47	 Loss: 0.1596	 Accuracy: 0.9432
# 100%|██████████| 1500/1500 [13:08<00:00,  1.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 48	 Loss: 0.1581	 Accuracy: 0.9426
# 100%|██████████| 1500/1500 [13:05<00:00,  1.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 49	 Loss: 0.1565	 Accuracy: 0.9377
# 100%|██████████| 1500/1500 [13:01<00:00,  1.92it/s]
# Model: Kmeans 	 Epoch: 50	 Loss: 0.1563	 Accuracy: 0.9430
# Save plot as ./alpha_value/alpha_value_1.0.png
# Model 1 name: TeacherModelCNN
# Model: TeacherModel	 Epoch: 1	 Loss: 0.4987	 Accuracy: 0.9257
# Model: TeacherModel	 Epoch: 2	 Loss: 0.2605	 Accuracy: 0.9443
# Model: TeacherModel	 Epoch: 3	 Loss: 0.2059	 Accuracy: 0.9565
# Model: TeacherModel	 Epoch: 4	 Loss: 0.1803	 Accuracy: 0.9604
# Model: TeacherModel	 Epoch: 5	 Loss: 0.1632	 Accuracy: 0.9612
# Model: TeacherModel	 Epoch: 6	 Loss: 0.1472	 Accuracy: 0.9670
# Model: TeacherModel	 Epoch: 7	 Loss: 0.1372	 Accuracy: 0.9680
# Model: TeacherModel	 Epoch: 8	 Loss: 0.1296	 Accuracy: 0.9701
# Model: TeacherModel	 Epoch: 9	 Loss: 0.1229	 Accuracy: 0.9712
# Model: TeacherModel	 Epoch: 10	 Loss: 0.1158	 Accuracy: 0.9718
# Model: TeacherModel	 Epoch: 11	 Loss: 0.1111	 Accuracy: 0.9710
# Model: TeacherModel	 Epoch: 12	 Loss: 0.1035	 Accuracy: 0.9720
# Model: TeacherModel	 Epoch: 13	 Loss: 0.1008	 Accuracy: 0.9742
# Model: TeacherModel	 Epoch: 14	 Loss: 0.0970	 Accuracy: 0.9733
# Model: TeacherModel	 Epoch: 15	 Loss: 0.0944	 Accuracy: 0.9748
# Model: TeacherModel	 Epoch: 16	 Loss: 0.0888	 Accuracy: 0.9754
# Model: TeacherModel	 Epoch: 17	 Loss: 0.0858	 Accuracy: 0.9757
# Model: TeacherModel	 Epoch: 18	 Loss: 0.0828	 Accuracy: 0.9750
# Model: TeacherModel	 Epoch: 19	 Loss: 0.0817	 Accuracy: 0.9762
# Model: TeacherModel	 Epoch: 20	 Loss: 0.0825	 Accuracy: 0.9755
# Model: TeacherModel	 Epoch: 21	 Loss: 0.0760	 Accuracy: 0.9792
# Model: TeacherModel	 Epoch: 22	 Loss: 0.0764	 Accuracy: 0.9768
# Model: TeacherModel	 Epoch: 23	 Loss: 0.0745	 Accuracy: 0.9766
# Model: TeacherModel	 Epoch: 24	 Loss: 0.0764	 Accuracy: 0.9787
# Model: TeacherModel	 Epoch: 25	 Loss: 0.0715	 Accuracy: 0.9782
# Model: TeacherModel	 Epoch: 26	 Loss: 0.0697	 Accuracy: 0.9800
# Model: TeacherModel	 Epoch: 27	 Loss: 0.0680	 Accuracy: 0.9792
# Model: TeacherModel	 Epoch: 28	 Loss: 0.0665	 Accuracy: 0.9773
# Model: TeacherModel	 Epoch: 29	 Loss: 0.0611	 Accuracy: 0.9784
# Model: TeacherModel	 Epoch: 30	 Loss: 0.0644	 Accuracy: 0.9788
# Model: TeacherModel	 Epoch: 31	 Loss: 0.0615	 Accuracy: 0.9803
# Model: TeacherModel	 Epoch: 32	 Loss: 0.0612	 Accuracy: 0.9810
# Model: TeacherModel	 Epoch: 33	 Loss: 0.0578	 Accuracy: 0.9806
# Model: TeacherModel	 Epoch: 34	 Loss: 0.0606	 Accuracy: 0.9786
# Model: TeacherModel	 Epoch: 35	 Loss: 0.0602	 Accuracy: 0.9812
# Model: TeacherModel	 Epoch: 36	 Loss: 0.0554	 Accuracy: 0.9790
# Model: TeacherModel	 Epoch: 37	 Loss: 0.0548	 Accuracy: 0.9812
# Model: TeacherModel	 Epoch: 38	 Loss: 0.0571	 Accuracy: 0.9818
# Model: TeacherModel	 Epoch: 39	 Loss: 0.0540	 Accuracy: 0.9796
# Model: TeacherModel	 Epoch: 40	 Loss: 0.0522	 Accuracy: 0.9822
# Model: TeacherModel	 Epoch: 41	 Loss: 0.0548	 Accuracy: 0.9818
# Model: TeacherModel	 Epoch: 42	 Loss: 0.0524	 Accuracy: 0.9813
# Model: TeacherModel	 Epoch: 43	 Loss: 0.0532	 Accuracy: 0.9824
# Model: TeacherModel	 Epoch: 44	 Loss: 0.0512	 Accuracy: 0.9824
# Model: TeacherModel	 Epoch: 45	 Loss: 0.0509	 Accuracy: 0.9804
# Model: TeacherModel	 Epoch: 46	 Loss: 0.0493	 Accuracy: 0.9805
# Model: TeacherModel	 Epoch: 47	 Loss: 0.0496	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 48	 Loss: 0.0499	 Accuracy: 0.9814
# Model: TeacherModel	 Epoch: 49	 Loss: 0.0477	 Accuracy: 0.9823
# Model: TeacherModel	 Epoch: 50	 Loss: 0.0485	 Accuracy: 0.9826
# Model 2 name: StudentModelA
# Model: StudentModel	 Epoch: 1	 Loss: 0.8593	 Accuracy: 0.8705
# Model: StudentModel	 Epoch: 2	 Loss: 0.4074	 Accuracy: 0.8902
# Model: StudentModel	 Epoch: 3	 Loss: 0.3536	 Accuracy: 0.9010
# Model: StudentModel	 Epoch: 4	 Loss: 0.3241	 Accuracy: 0.9051
# Model: StudentModel	 Epoch: 5	 Loss: 0.3038	 Accuracy: 0.9104
# Model: StudentModel	 Epoch: 6	 Loss: 0.2867	 Accuracy: 0.9166
# Model: StudentModel	 Epoch: 7	 Loss: 0.2747	 Accuracy: 0.9147
# Model: StudentModel	 Epoch: 8	 Loss: 0.2635	 Accuracy: 0.9201
# Model: StudentModel	 Epoch: 9	 Loss: 0.2546	 Accuracy: 0.9226
# Model: StudentModel	 Epoch: 10	 Loss: 0.2479	 Accuracy: 0.9256
# Model: StudentModel	 Epoch: 11	 Loss: 0.2410	 Accuracy: 0.9246
# Model: StudentModel	 Epoch: 12	 Loss: 0.2347	 Accuracy: 0.9273
# Model: StudentModel	 Epoch: 13	 Loss: 0.2294	 Accuracy: 0.9277
# Model: StudentModel	 Epoch: 14	 Loss: 0.2239	 Accuracy: 0.9309
# Model: StudentModel	 Epoch: 15	 Loss: 0.2193	 Accuracy: 0.9318
# Model: StudentModel	 Epoch: 16	 Loss: 0.2154	 Accuracy: 0.9313
# Model: StudentModel	 Epoch: 17	 Loss: 0.2115	 Accuracy: 0.9308
# Model: StudentModel	 Epoch: 18	 Loss: 0.2080	 Accuracy: 0.9314
# Model: StudentModel	 Epoch: 19	 Loss: 0.2041	 Accuracy: 0.9333
# Model: StudentModel	 Epoch: 20	 Loss: 0.2015	 Accuracy: 0.9326
# Model: StudentModel	 Epoch: 21	 Loss: 0.1978	 Accuracy: 0.9303
# Model: StudentModel	 Epoch: 22	 Loss: 0.1963	 Accuracy: 0.9324
# Model: StudentModel	 Epoch: 23	 Loss: 0.1931	 Accuracy: 0.9331
# Model: StudentModel	 Epoch: 24	 Loss: 0.1912	 Accuracy: 0.9376
# Model: StudentModel	 Epoch: 25	 Loss: 0.1880	 Accuracy: 0.9337
# Model: StudentModel	 Epoch: 26	 Loss: 0.1862	 Accuracy: 0.9313
# Model: StudentModel	 Epoch: 27	 Loss: 0.1829	 Accuracy: 0.9366
# Model: StudentModel	 Epoch: 28	 Loss: 0.1811	 Accuracy: 0.9368
# Model: StudentModel	 Epoch: 29	 Loss: 0.1798	 Accuracy: 0.9363
# Model: StudentModel	 Epoch: 30	 Loss: 0.1771	 Accuracy: 0.9387
# Model: StudentModel	 Epoch: 31	 Loss: 0.1749	 Accuracy: 0.9366
# Model: StudentModel	 Epoch: 32	 Loss: 0.1726	 Accuracy: 0.9389
# Model: StudentModel	 Epoch: 33	 Loss: 0.1708	 Accuracy: 0.9402
# Model: StudentModel	 Epoch: 34	 Loss: 0.1691	 Accuracy: 0.9391
# Model: StudentModel	 Epoch: 35	 Loss: 0.1678	 Accuracy: 0.9377
# Model: StudentModel	 Epoch: 36	 Loss: 0.1656	 Accuracy: 0.9373
# Model: StudentModel	 Epoch: 37	 Loss: 0.1640	 Accuracy: 0.9388
# Model: StudentModel	 Epoch: 38	 Loss: 0.1619	 Accuracy: 0.9358
# Model: StudentModel	 Epoch: 39	 Loss: 0.1602	 Accuracy: 0.9381
# Model: StudentModel	 Epoch: 40	 Loss: 0.1588	 Accuracy: 0.9388
# Model: StudentModel	 Epoch: 41	 Loss: 0.1563	 Accuracy: 0.9377
# Model: StudentModel	 Epoch: 42	 Loss: 0.1555	 Accuracy: 0.9403
# Model: StudentModel	 Epoch: 43	 Loss: 0.1537	 Accuracy: 0.9422
# Model: StudentModel	 Epoch: 44	 Loss: 0.1521	 Accuracy: 0.9426
# Model: StudentModel	 Epoch: 45	 Loss: 0.1510	 Accuracy: 0.9397
# Model: StudentModel	 Epoch: 46	 Loss: 0.1498	 Accuracy: 0.9416
# Model: StudentModel	 Epoch: 47	 Loss: 0.1476	 Accuracy: 0.9417
# Model: StudentModel	 Epoch: 48	 Loss: 0.1467	 Accuracy: 0.9435
# Model: StudentModel	 Epoch: 49	 Loss: 0.1454	 Accuracy: 0.9410
# Model: StudentModel	 Epoch: 50	 Loss: 0.1438	 Accuracy: 0.9428
# Model 3 name: StudentModelB
# Model: StudentModel	 Epoch: 1	 Loss: 25.5018	 Accuracy: 0.8488
# Model: StudentModel	 Epoch: 2	 Loss: 20.7298	 Accuracy: 0.8812
# Model: StudentModel	 Epoch: 3	 Loss: 20.1546	 Accuracy: 0.8934
# Model: StudentModel	 Epoch: 4	 Loss: 19.9399	 Accuracy: 0.8983
# Model: StudentModel	 Epoch: 5	 Loss: 19.8250	 Accuracy: 0.9032
# Model: StudentModel	 Epoch: 6	 Loss: 19.7392	 Accuracy: 0.9038
# Model: StudentModel	 Epoch: 7	 Loss: 19.6810	 Accuracy: 0.9065
# Model: StudentModel	 Epoch: 8	 Loss: 19.6272	 Accuracy: 0.9095
# Model: StudentModel	 Epoch: 9	 Loss: 19.5847	 Accuracy: 0.9116
# Model: StudentModel	 Epoch: 10	 Loss: 19.5443	 Accuracy: 0.9093
# Model: StudentModel	 Epoch: 11	 Loss: 19.5079	 Accuracy: 0.9141
# Model: StudentModel	 Epoch: 12	 Loss: 19.4743	 Accuracy: 0.9148
# Model: StudentModel	 Epoch: 13	 Loss: 19.4422	 Accuracy: 0.9140
# Model: StudentModel	 Epoch: 14	 Loss: 19.4162	 Accuracy: 0.9161
# Model: StudentModel	 Epoch: 15	 Loss: 19.3820	 Accuracy: 0.9162
# Model: StudentModel	 Epoch: 16	 Loss: 19.3623	 Accuracy: 0.9181
# Model: StudentModel	 Epoch: 17	 Loss: 19.3356	 Accuracy: 0.9185
# Model: StudentModel	 Epoch: 18	 Loss: 19.3127	 Accuracy: 0.9177
# Model: StudentModel	 Epoch: 19	 Loss: 19.2943	 Accuracy: 0.9195
# Model: StudentModel	 Epoch: 20	 Loss: 19.2709	 Accuracy: 0.9186
# Model: StudentModel	 Epoch: 21	 Loss: 19.2496	 Accuracy: 0.9206
# Model: StudentModel	 Epoch: 22	 Loss: 19.2336	 Accuracy: 0.9213
# Model: StudentModel	 Epoch: 23	 Loss: 19.2150	 Accuracy: 0.9220
# Model: StudentModel	 Epoch: 24	 Loss: 19.1942	 Accuracy: 0.9214
# Model: StudentModel	 Epoch: 25	 Loss: 19.1805	 Accuracy: 0.9213
# Model: StudentModel	 Epoch: 26	 Loss: 19.1626	 Accuracy: 0.9216
# Model: StudentModel	 Epoch: 27	 Loss: 19.1465	 Accuracy: 0.9232
# Model: StudentModel	 Epoch: 28	 Loss: 19.1310	 Accuracy: 0.9242
# Model: StudentModel	 Epoch: 29	 Loss: 19.1158	 Accuracy: 0.9230
# Model: StudentModel	 Epoch: 30	 Loss: 19.0980	 Accuracy: 0.9237
# Model: StudentModel	 Epoch: 31	 Loss: 19.0808	 Accuracy: 0.9238
# Model: StudentModel	 Epoch: 32	 Loss: 19.0660	 Accuracy: 0.9240
# Model: StudentModel	 Epoch: 33	 Loss: 19.0537	 Accuracy: 0.9253
# Model: StudentModel	 Epoch: 34	 Loss: 19.0346	 Accuracy: 0.9257
# Model: StudentModel	 Epoch: 35	 Loss: 19.0219	 Accuracy: 0.9263
# Model: StudentModel	 Epoch: 36	 Loss: 19.0056	 Accuracy: 0.9277
# Model: StudentModel	 Epoch: 37	 Loss: 18.9969	 Accuracy: 0.9277
# Model: StudentModel	 Epoch: 38	 Loss: 18.9845	 Accuracy: 0.9260
# Model: StudentModel	 Epoch: 39	 Loss: 18.9697	 Accuracy: 0.9277
# Model: StudentModel	 Epoch: 40	 Loss: 18.9640	 Accuracy: 0.9283
# Model: StudentModel	 Epoch: 41	 Loss: 18.9521	 Accuracy: 0.9273
# Model: StudentModel	 Epoch: 42	 Loss: 18.9463	 Accuracy: 0.9306
# Model: StudentModel	 Epoch: 43	 Loss: 18.9347	 Accuracy: 0.9320
# Model: StudentModel	 Epoch: 44	 Loss: 18.9296	 Accuracy: 0.9319
# Model: StudentModel	 Epoch: 45	 Loss: 18.9203	 Accuracy: 0.9305
# Model: StudentModel	 Epoch: 46	 Loss: 18.9136	 Accuracy: 0.9320
# Model: StudentModel	 Epoch: 47	 Loss: 18.9075	 Accuracy: 0.9323
# Model: StudentModel	 Epoch: 48	 Loss: 18.8990	 Accuracy: 0.9321
# Model: StudentModel	 Epoch: 49	 Loss: 18.8946	 Accuracy: 0.9336
# Model: StudentModel	 Epoch: 50	 Loss: 18.8890	 Accuracy: 0.9311
# Model 4 name: StudentModelC
# 100%|██████████| 1500/1500 [12:46<00:00,  1.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 1	 Loss: 1.0489	 Accuracy: 0.8598
# 100%|██████████| 1500/1500 [12:16<00:00,  2.04it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 2	 Loss: 0.6711	 Accuracy: 0.8862
# 100%|██████████| 1500/1500 [12:21<00:00,  2.02it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 3	 Loss: 0.6121	 Accuracy: 0.8968
# 100%|██████████| 1500/1500 [13:03<00:00,  1.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 4	 Loss: 0.5829	 Accuracy: 0.9005
# 100%|██████████| 1500/1500 [13:08<00:00,  1.90it/s]
# Model: Kmeans 	 Epoch: 5	 Loss: 0.5611	 Accuracy: 0.9048
# 100%|██████████| 1500/1500 [12:51<00:00,  1.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 6	 Loss: 0.5440	 Accuracy: 0.9093
# 100%|██████████| 1500/1500 [12:45<00:00,  1.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 7	 Loss: 0.5328	 Accuracy: 0.9107
# 100%|██████████| 1500/1500 [12:46<00:00,  1.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 8	 Loss: 0.5233	 Accuracy: 0.9145
# 100%|██████████| 1500/1500 [12:36<00:00,  1.98it/s]
# Model: Kmeans 	 Epoch: 9	 Loss: 0.5166	 Accuracy: 0.9116
# 100%|██████████| 1500/1500 [12:11<00:00,  2.05it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 10	 Loss: 0.5105	 Accuracy: 0.9147
# 100%|██████████| 1500/1500 [12:26<00:00,  2.01it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 11	 Loss: 0.5048	 Accuracy: 0.9190
# 100%|██████████| 1500/1500 [12:53<00:00,  1.94it/s]
# Model: Kmeans 	 Epoch: 12	 Loss: 0.5007	 Accuracy: 0.9172
# 100%|██████████| 1500/1500 [12:50<00:00,  1.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 13	 Loss: 0.4970	 Accuracy: 0.9174
# 100%|██████████| 1500/1500 [12:52<00:00,  1.94it/s]
# Model: Kmeans 	 Epoch: 14	 Loss: 0.4938	 Accuracy: 0.9165
# 100%|██████████| 1500/1500 [12:51<00:00,  1.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 15	 Loss: 0.4901	 Accuracy: 0.9207
# 100%|██████████| 1500/1500 [12:29<00:00,  2.00it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 16	 Loss: 0.4872	 Accuracy: 0.9193
# 100%|██████████| 1500/1500 [12:36<00:00,  1.98it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 17	 Loss: 0.4845	 Accuracy: 0.9221
# 100%|██████████| 1500/1500 [12:32<00:00,  1.99it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 18	 Loss: 0.4828	 Accuracy: 0.9213
# 100%|██████████| 1500/1500 [13:04<00:00,  1.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 19	 Loss: 0.4788	 Accuracy: 0.9227
# 100%|██████████| 1500/1500 [12:48<00:00,  1.95it/s]
# Model: Kmeans 	 Epoch: 20	 Loss: 0.4774	 Accuracy: 0.9220
# 100%|██████████| 1500/1500 [12:26<00:00,  2.01it/s]
# Model: Kmeans 	 Epoch: 21	 Loss: 0.4742	 Accuracy: 0.9218
# 100%|██████████| 1500/1500 [12:38<00:00,  1.98it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 22	 Loss: 0.4726	 Accuracy: 0.9232
# 100%|██████████| 1500/1500 [12:40<00:00,  1.97it/s]
# Model: Kmeans 	 Epoch: 23	 Loss: 0.4700	 Accuracy: 0.9207
# 100%|██████████| 1500/1500 [12:57<00:00,  1.93it/s]
# Model: Kmeans 	 Epoch: 24	 Loss: 0.4681	 Accuracy: 0.9246
# 100%|██████████| 1500/1500 [12:52<00:00,  1.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 25	 Loss: 0.4664	 Accuracy: 0.9239
# 100%|██████████| 1500/1500 [12:23<00:00,  2.02it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 26	 Loss: 0.4642	 Accuracy: 0.9251
# 100%|██████████| 1500/1500 [12:57<00:00,  1.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 27	 Loss: 0.4625	 Accuracy: 0.9252
# 100%|██████████| 1500/1500 [12:45<00:00,  1.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 28	 Loss: 0.4604	 Accuracy: 0.9263
# 100%|██████████| 1500/1500 [12:41<00:00,  1.97it/s]
# Model: Kmeans 	 Epoch: 29	 Loss: 0.4579	 Accuracy: 0.9259
# 100%|██████████| 1500/1500 [12:57<00:00,  1.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 30	 Loss: 0.4564	 Accuracy: 0.9263
# 100%|██████████| 1500/1500 [12:49<00:00,  1.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 31	 Loss: 0.4546	 Accuracy: 0.9264
# 100%|██████████| 1500/1500 [12:58<00:00,  1.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 32	 Loss: 0.4533	 Accuracy: 0.9277
# 100%|██████████| 1500/1500 [12:58<00:00,  1.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 33	 Loss: 0.4511	 Accuracy: 0.9251
# 100%|██████████| 1500/1500 [12:29<00:00,  2.00it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 34	 Loss: 0.4502	 Accuracy: 0.9288
# 100%|██████████| 1500/1500 [12:32<00:00,  1.99it/s]
# Model: Kmeans 	 Epoch: 35	 Loss: 0.4485	 Accuracy: 0.9301
# 100%|██████████| 1500/1500 [12:33<00:00,  1.99it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 36	 Loss: 0.4467	 Accuracy: 0.9297
# 100%|██████████| 1500/1500 [12:25<00:00,  2.01it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 37	 Loss: 0.4450	 Accuracy: 0.9303
# 100%|██████████| 1500/1500 [12:39<00:00,  1.97it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 38	 Loss: 0.4434	 Accuracy: 0.9312
# 100%|██████████| 1500/1500 [12:35<00:00,  1.99it/s]
# Model: Kmeans 	 Epoch: 39	 Loss: 0.4423	 Accuracy: 0.9308
# 100%|██████████| 1500/1500 [12:44<00:00,  1.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 40	 Loss: 0.4402	 Accuracy: 0.9293
# 100%|██████████| 1500/1500 [12:46<00:00,  1.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 41	 Loss: 0.4390	 Accuracy: 0.9314
# 100%|██████████| 1500/1500 [12:44<00:00,  1.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 42	 Loss: 0.4382	 Accuracy: 0.9330
# 100%|██████████| 1500/1500 [12:55<00:00,  1.93it/s]
# Model: Kmeans 	 Epoch: 43	 Loss: 0.4364	 Accuracy: 0.9326
# 100%|██████████| 1500/1500 [12:48<00:00,  1.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 44	 Loss: 0.4354	 Accuracy: 0.9317
# 100%|██████████| 1500/1500 [12:35<00:00,  1.99it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 45	 Loss: 0.4341	 Accuracy: 0.9333
# 100%|██████████| 1500/1500 [12:54<00:00,  1.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 46	 Loss: 0.4332	 Accuracy: 0.9307
# 100%|██████████| 1500/1500 [12:28<00:00,  2.00it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 47	 Loss: 0.4311	 Accuracy: 0.9344
# 100%|██████████| 1500/1500 [12:17<00:00,  2.04it/s]
# Model: Kmeans 	 Epoch: 48	 Loss: 0.4303	 Accuracy: 0.9308
# 100%|██████████| 1500/1500 [12:49<00:00,  1.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 49	 Loss: 0.4295	 Accuracy: 0.9317
# 100%|██████████| 1500/1500 [12:41<00:00,  1.97it/s]
# Model: Kmeans 	 Epoch: 50	 Loss: 0.4285	 Accuracy: 0.9348
# Save plot as ./alpha_value/alpha_value_0.9.png
# Model 1 name: TeacherModelCNN
# Model: TeacherModel	 Epoch: 1	 Loss: 0.4947	 Accuracy: 0.9243
# Model: TeacherModel	 Epoch: 2	 Loss: 0.2585	 Accuracy: 0.9375
# Model: TeacherModel	 Epoch: 3	 Loss: 0.2076	 Accuracy: 0.9517
# Model: TeacherModel	 Epoch: 4	 Loss: 0.1805	 Accuracy: 0.9581
# Model: TeacherModel	 Epoch: 5	 Loss: 0.1603	 Accuracy: 0.9644
# Model: TeacherModel	 Epoch: 6	 Loss: 0.1508	 Accuracy: 0.9687
# Model: TeacherModel	 Epoch: 7	 Loss: 0.1360	 Accuracy: 0.9660
# Model: TeacherModel	 Epoch: 8	 Loss: 0.1300	 Accuracy: 0.9696
# Model: TeacherModel	 Epoch: 9	 Loss: 0.1223	 Accuracy: 0.9677
# Model: TeacherModel	 Epoch: 10	 Loss: 0.1159	 Accuracy: 0.9728
# Model: TeacherModel	 Epoch: 11	 Loss: 0.1085	 Accuracy: 0.9725
# Model: TeacherModel	 Epoch: 12	 Loss: 0.1055	 Accuracy: 0.9741
# Model: TeacherModel	 Epoch: 13	 Loss: 0.1008	 Accuracy: 0.9725
# Model: TeacherModel	 Epoch: 14	 Loss: 0.0979	 Accuracy: 0.9752
# Model: TeacherModel	 Epoch: 15	 Loss: 0.0941	 Accuracy: 0.9752
# Model: TeacherModel	 Epoch: 16	 Loss: 0.0933	 Accuracy: 0.9748
# Model: TeacherModel	 Epoch: 17	 Loss: 0.0870	 Accuracy: 0.9747
# Model: TeacherModel	 Epoch: 18	 Loss: 0.0830	 Accuracy: 0.9768
# Model: TeacherModel	 Epoch: 19	 Loss: 0.0854	 Accuracy: 0.9748
# Model: TeacherModel	 Epoch: 20	 Loss: 0.0830	 Accuracy: 0.9764
# Model: TeacherModel	 Epoch: 21	 Loss: 0.0790	 Accuracy: 0.9753
# Model: TeacherModel	 Epoch: 22	 Loss: 0.0769	 Accuracy: 0.9799
# Model: TeacherModel	 Epoch: 23	 Loss: 0.0733	 Accuracy: 0.9775
# Model: TeacherModel	 Epoch: 24	 Loss: 0.0737	 Accuracy: 0.9785
# Model: TeacherModel	 Epoch: 25	 Loss: 0.0691	 Accuracy: 0.9798
# Model: TeacherModel	 Epoch: 26	 Loss: 0.0690	 Accuracy: 0.9812
# Model: TeacherModel	 Epoch: 27	 Loss: 0.0695	 Accuracy: 0.9794
# Model: TeacherModel	 Epoch: 28	 Loss: 0.0676	 Accuracy: 0.9789
# Model: TeacherModel	 Epoch: 29	 Loss: 0.0636	 Accuracy: 0.9800
# Model: TeacherModel	 Epoch: 30	 Loss: 0.0633	 Accuracy: 0.9789
# Model: TeacherModel	 Epoch: 31	 Loss: 0.0621	 Accuracy: 0.9794
# Model: TeacherModel	 Epoch: 32	 Loss: 0.0610	 Accuracy: 0.9810
# Model: TeacherModel	 Epoch: 33	 Loss: 0.0619	 Accuracy: 0.9777
# Model: TeacherModel	 Epoch: 34	 Loss: 0.0592	 Accuracy: 0.9805
# Model: TeacherModel	 Epoch: 35	 Loss: 0.0602	 Accuracy: 0.9804
# Model: TeacherModel	 Epoch: 36	 Loss: 0.0584	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 37	 Loss: 0.0606	 Accuracy: 0.9815
# Model: TeacherModel	 Epoch: 38	 Loss: 0.0570	 Accuracy: 0.9817
# Model: TeacherModel	 Epoch: 39	 Loss: 0.0560	 Accuracy: 0.9804
# Model: TeacherModel	 Epoch: 40	 Loss: 0.0562	 Accuracy: 0.9804
# Model: TeacherModel	 Epoch: 41	 Loss: 0.0531	 Accuracy: 0.9824
# Model: TeacherModel	 Epoch: 42	 Loss: 0.0526	 Accuracy: 0.9800
# Model: TeacherModel	 Epoch: 43	 Loss: 0.0532	 Accuracy: 0.9811
# Model: TeacherModel	 Epoch: 44	 Loss: 0.0516	 Accuracy: 0.9827
# Model: TeacherModel	 Epoch: 45	 Loss: 0.0499	 Accuracy: 0.9818
# Model: TeacherModel	 Epoch: 46	 Loss: 0.0497	 Accuracy: 0.9827
# Model: TeacherModel	 Epoch: 47	 Loss: 0.0514	 Accuracy: 0.9809
# Model: TeacherModel	 Epoch: 48	 Loss: 0.0491	 Accuracy: 0.9815
# Model: TeacherModel	 Epoch: 49	 Loss: 0.0481	 Accuracy: 0.9830
# Model: TeacherModel	 Epoch: 50	 Loss: 0.0502	 Accuracy: 0.9818
# Model 2 name: StudentModelA
# Model: StudentModel	 Epoch: 1	 Loss: 0.9361	 Accuracy: 0.8658
# Model: StudentModel	 Epoch: 2	 Loss: 0.4185	 Accuracy: 0.8928
# Model: StudentModel	 Epoch: 3	 Loss: 0.3536	 Accuracy: 0.9006
# Model: StudentModel	 Epoch: 4	 Loss: 0.3272	 Accuracy: 0.9031
# Model: StudentModel	 Epoch: 5	 Loss: 0.3123	 Accuracy: 0.9073
# Model: StudentModel	 Epoch: 6	 Loss: 0.3021	 Accuracy: 0.9108
# Model: StudentModel	 Epoch: 7	 Loss: 0.2933	 Accuracy: 0.9097
# Model: StudentModel	 Epoch: 8	 Loss: 0.2864	 Accuracy: 0.9138
# Model: StudentModel	 Epoch: 9	 Loss: 0.2801	 Accuracy: 0.9158
# Model: StudentModel	 Epoch: 10	 Loss: 0.2737	 Accuracy: 0.9173
# Model: StudentModel	 Epoch: 11	 Loss: 0.2679	 Accuracy: 0.9157
# Model: StudentModel	 Epoch: 12	 Loss: 0.2631	 Accuracy: 0.9171
# Model: StudentModel	 Epoch: 13	 Loss: 0.2580	 Accuracy: 0.9217
# Model: StudentModel	 Epoch: 14	 Loss: 0.2530	 Accuracy: 0.9204
# Model: StudentModel	 Epoch: 15	 Loss: 0.2493	 Accuracy: 0.9205
# Model: StudentModel	 Epoch: 16	 Loss: 0.2445	 Accuracy: 0.9226
# Model: StudentModel	 Epoch: 17	 Loss: 0.2409	 Accuracy: 0.9240
# Model: StudentModel	 Epoch: 18	 Loss: 0.2375	 Accuracy: 0.9263
# Model: StudentModel	 Epoch: 19	 Loss: 0.2331	 Accuracy: 0.9278
# Model: StudentModel	 Epoch: 20	 Loss: 0.2303	 Accuracy: 0.9237
# Model: StudentModel	 Epoch: 21	 Loss: 0.2262	 Accuracy: 0.9286
# Model: StudentModel	 Epoch: 22	 Loss: 0.2239	 Accuracy: 0.9268
# Model: StudentModel	 Epoch: 23	 Loss: 0.2208	 Accuracy: 0.9288
# Model: StudentModel	 Epoch: 24	 Loss: 0.2188	 Accuracy: 0.9287
# Model: StudentModel	 Epoch: 25	 Loss: 0.2152	 Accuracy: 0.9303
# Model: StudentModel	 Epoch: 26	 Loss: 0.2129	 Accuracy: 0.9308
# Model: StudentModel	 Epoch: 27	 Loss: 0.2105	 Accuracy: 0.9307
# Model: StudentModel	 Epoch: 28	 Loss: 0.2080	 Accuracy: 0.9293
# Model: StudentModel	 Epoch: 29	 Loss: 0.2063	 Accuracy: 0.9322
# Model: StudentModel	 Epoch: 30	 Loss: 0.2052	 Accuracy: 0.9323
# Model: StudentModel	 Epoch: 31	 Loss: 0.2018	 Accuracy: 0.9293
# Model: StudentModel	 Epoch: 32	 Loss: 0.2002	 Accuracy: 0.9315
# Model: StudentModel	 Epoch: 33	 Loss: 0.1987	 Accuracy: 0.9334
# Model: StudentModel	 Epoch: 34	 Loss: 0.1968	 Accuracy: 0.9334
# Model: StudentModel	 Epoch: 35	 Loss: 0.1942	 Accuracy: 0.9317
# Model: StudentModel	 Epoch: 36	 Loss: 0.1929	 Accuracy: 0.9342
# Model: StudentModel	 Epoch: 37	 Loss: 0.1909	 Accuracy: 0.9323
# Model: StudentModel	 Epoch: 38	 Loss: 0.1896	 Accuracy: 0.9360
# Model: StudentModel	 Epoch: 39	 Loss: 0.1880	 Accuracy: 0.9341
# Model: StudentModel	 Epoch: 40	 Loss: 0.1866	 Accuracy: 0.9368
# Model: StudentModel	 Epoch: 41	 Loss: 0.1850	 Accuracy: 0.9357
# Model: StudentModel	 Epoch: 42	 Loss: 0.1840	 Accuracy: 0.9359
# Model: StudentModel	 Epoch: 43	 Loss: 0.1820	 Accuracy: 0.9373
# Model: StudentModel	 Epoch: 44	 Loss: 0.1812	 Accuracy: 0.9384
# Model: StudentModel	 Epoch: 45	 Loss: 0.1796	 Accuracy: 0.9346
# Model: StudentModel	 Epoch: 46	 Loss: 0.1783	 Accuracy: 0.9358
# Model: StudentModel	 Epoch: 47	 Loss: 0.1772	 Accuracy: 0.9383
# Model: StudentModel	 Epoch: 48	 Loss: 0.1756	 Accuracy: 0.9393
# Model: StudentModel	 Epoch: 49	 Loss: 0.1736	 Accuracy: 0.9389
# Model: StudentModel	 Epoch: 50	 Loss: 0.1734	 Accuracy: 0.9370
# Model 3 name: StudentModelB
# Model: StudentModel	 Epoch: 1	 Loss: 24.9044	 Accuracy: 0.8455
# Model: StudentModel	 Epoch: 2	 Loss: 20.7096	 Accuracy: 0.8791
# Model: StudentModel	 Epoch: 3	 Loss: 20.1792	 Accuracy: 0.8921
# Model: StudentModel	 Epoch: 4	 Loss: 19.9100	 Accuracy: 0.8955
# Model: StudentModel	 Epoch: 5	 Loss: 19.7423	 Accuracy: 0.9049
# Model: StudentModel	 Epoch: 6	 Loss: 19.6252	 Accuracy: 0.9075
# Model: StudentModel	 Epoch: 7	 Loss: 19.5357	 Accuracy: 0.9098
# Model: StudentModel	 Epoch: 8	 Loss: 19.4626	 Accuracy: 0.9131
# Model: StudentModel	 Epoch: 9	 Loss: 19.3985	 Accuracy: 0.9163
# Model: StudentModel	 Epoch: 10	 Loss: 19.3430	 Accuracy: 0.9166
# Model: StudentModel	 Epoch: 11	 Loss: 19.2953	 Accuracy: 0.9223
# Model: StudentModel	 Epoch: 12	 Loss: 19.2523	 Accuracy: 0.9224
# Model: StudentModel	 Epoch: 13	 Loss: 19.2160	 Accuracy: 0.9234
# Model: StudentModel	 Epoch: 14	 Loss: 19.1816	 Accuracy: 0.9247
# Model: StudentModel	 Epoch: 15	 Loss: 19.1495	 Accuracy: 0.9263
# Model: StudentModel	 Epoch: 16	 Loss: 19.1190	 Accuracy: 0.9283
# Model: StudentModel	 Epoch: 17	 Loss: 19.0945	 Accuracy: 0.9266
# Model: StudentModel	 Epoch: 18	 Loss: 19.0647	 Accuracy: 0.9287
# Model: StudentModel	 Epoch: 19	 Loss: 19.0458	 Accuracy: 0.9289
# Model: StudentModel	 Epoch: 20	 Loss: 19.0222	 Accuracy: 0.9312
# Model: StudentModel	 Epoch: 21	 Loss: 18.9999	 Accuracy: 0.9319
# Model: StudentModel	 Epoch: 22	 Loss: 18.9786	 Accuracy: 0.9312
# Model: StudentModel	 Epoch: 23	 Loss: 18.9632	 Accuracy: 0.9333
# Model: StudentModel	 Epoch: 24	 Loss: 18.9424	 Accuracy: 0.9327
# Model: StudentModel	 Epoch: 25	 Loss: 18.9250	 Accuracy: 0.9331
# Model: StudentModel	 Epoch: 26	 Loss: 18.9140	 Accuracy: 0.9333
# Model: StudentModel	 Epoch: 27	 Loss: 18.8943	 Accuracy: 0.9352
# Model: StudentModel	 Epoch: 28	 Loss: 18.8831	 Accuracy: 0.9361
# Model: StudentModel	 Epoch: 29	 Loss: 18.8684	 Accuracy: 0.9363
# Model: StudentModel	 Epoch: 30	 Loss: 18.8579	 Accuracy: 0.9363
# Model: StudentModel	 Epoch: 31	 Loss: 18.8458	 Accuracy: 0.9369
# Model: StudentModel	 Epoch: 32	 Loss: 18.8372	 Accuracy: 0.9382
# Model: StudentModel	 Epoch: 33	 Loss: 18.8269	 Accuracy: 0.9389
# Model: StudentModel	 Epoch: 34	 Loss: 18.8138	 Accuracy: 0.9387
# Model: StudentModel	 Epoch: 35	 Loss: 18.8046	 Accuracy: 0.9392
# Model: StudentModel	 Epoch: 36	 Loss: 18.7971	 Accuracy: 0.9407
# Model: StudentModel	 Epoch: 37	 Loss: 18.7869	 Accuracy: 0.9376
# Model: StudentModel	 Epoch: 38	 Loss: 18.7775	 Accuracy: 0.9397
# Model: StudentModel	 Epoch: 39	 Loss: 18.7717	 Accuracy: 0.9403
# Model: StudentModel	 Epoch: 40	 Loss: 18.7635	 Accuracy: 0.9413
# Model: StudentModel	 Epoch: 41	 Loss: 18.7561	 Accuracy: 0.9388
# Model: StudentModel	 Epoch: 42	 Loss: 18.7473	 Accuracy: 0.9416
# Model: StudentModel	 Epoch: 43	 Loss: 18.7362	 Accuracy: 0.9397
# Model: StudentModel	 Epoch: 44	 Loss: 18.7318	 Accuracy: 0.9432
# Model: StudentModel	 Epoch: 45	 Loss: 18.7253	 Accuracy: 0.9420
# Model: StudentModel	 Epoch: 46	 Loss: 18.7158	 Accuracy: 0.9420
# Model: StudentModel	 Epoch: 47	 Loss: 18.7087	 Accuracy: 0.9411
# Model: StudentModel	 Epoch: 48	 Loss: 18.7029	 Accuracy: 0.9424
# Model: StudentModel	 Epoch: 49	 Loss: 18.6953	 Accuracy: 0.9417
# Model: StudentModel	 Epoch: 50	 Loss: 18.6895	 Accuracy: 0.9398
# Model 4 name: StudentModelC
# 100%|██████████| 1500/1500 [08:50<00:00,  2.83it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 1	 Loss: 0.9916	 Accuracy: 0.8684
# 100%|██████████| 1500/1500 [11:31<00:00,  2.17it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 2	 Loss: 0.7073	 Accuracy: 0.8960
# 100%|██████████| 1500/1500 [11:17<00:00,  2.22it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 3	 Loss: 0.6627	 Accuracy: 0.9018
# 100%|██████████| 1500/1500 [09:23<00:00,  2.66it/s]
# Model: Kmeans 	 Epoch: 4	 Loss: 0.6371	 Accuracy: 0.9087
# 100%|██████████| 1500/1500 [09:06<00:00,  2.74it/s]
# Model: Kmeans 	 Epoch: 5	 Loss: 0.6197	 Accuracy: 0.9155
# 100%|██████████| 1500/1500 [10:47<00:00,  2.32it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 6	 Loss: 0.6074	 Accuracy: 0.9141
# 100%|██████████| 1500/1500 [10:17<00:00,  2.43it/s]
# Model: Kmeans 	 Epoch: 7	 Loss: 0.5987	 Accuracy: 0.9117
# 100%|██████████| 1500/1500 [09:44<00:00,  2.57it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 8	 Loss: 0.5912	 Accuracy: 0.9183
# 100%|██████████| 1500/1500 [09:39<00:00,  2.59it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 9	 Loss: 0.5847	 Accuracy: 0.9217
# 100%|██████████| 1500/1500 [09:18<00:00,  2.68it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 10	 Loss: 0.5788	 Accuracy: 0.9213
# 100%|██████████| 1500/1500 [09:08<00:00,  2.74it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 11	 Loss: 0.5733	 Accuracy: 0.9240
# 100%|██████████| 1500/1500 [08:51<00:00,  2.82it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 12	 Loss: 0.5679	 Accuracy: 0.9279
# 100%|██████████| 1500/1500 [08:50<00:00,  2.83it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 13	 Loss: 0.5636	 Accuracy: 0.9254
# 100%|██████████| 1500/1500 [09:40<00:00,  2.58it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 14	 Loss: 0.5600	 Accuracy: 0.9297
# 100%|██████████| 1500/1500 [09:09<00:00,  2.73it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 15	 Loss: 0.5561	 Accuracy: 0.9303
# 100%|██████████| 1500/1500 [09:28<00:00,  2.64it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 16	 Loss: 0.5525	 Accuracy: 0.9313
# 100%|██████████| 1500/1500 [09:20<00:00,  2.68it/s]
# Model: Kmeans 	 Epoch: 17	 Loss: 0.5492	 Accuracy: 0.9313
# 100%|██████████| 1500/1500 [09:30<00:00,  2.63it/s]
# Model: Kmeans 	 Epoch: 18	 Loss: 0.5459	 Accuracy: 0.9313
# 100%|██████████| 1500/1500 [09:31<00:00,  2.62it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 19	 Loss: 0.5437	 Accuracy: 0.9335
# 100%|██████████| 1500/1500 [09:00<00:00,  2.78it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 20	 Loss: 0.5407	 Accuracy: 0.9325
# 100%|██████████| 1500/1500 [08:44<00:00,  2.86it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 21	 Loss: 0.5384	 Accuracy: 0.9340
# 100%|██████████| 1500/1500 [08:48<00:00,  2.84it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 22	 Loss: 0.5368	 Accuracy: 0.9290
# 100%|██████████| 1500/1500 [09:32<00:00,  2.62it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 23	 Loss: 0.5351	 Accuracy: 0.9349
# 100%|██████████| 1500/1500 [10:15<00:00,  2.44it/s]
# Model: Kmeans 	 Epoch: 24	 Loss: 0.5328	 Accuracy: 0.9347
# 100%|██████████| 1500/1500 [10:03<00:00,  2.49it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 25	 Loss: 0.5315	 Accuracy: 0.9341
# 100%|██████████| 1500/1500 [10:06<00:00,  2.47it/s]
# Model: Kmeans 	 Epoch: 26	 Loss: 0.5299	 Accuracy: 0.9357
# 100%|██████████| 1500/1500 [09:45<00:00,  2.56it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 27	 Loss: 0.5283	 Accuracy: 0.9334
# 100%|██████████| 1500/1500 [11:01<00:00,  2.27it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 28	 Loss: 0.5272	 Accuracy: 0.9357
# 100%|██████████| 1500/1500 [09:23<00:00,  2.66it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 29	 Loss: 0.5257	 Accuracy: 0.9375
# 100%|██████████| 1500/1500 [09:35<00:00,  2.61it/s]
# Model: Kmeans 	 Epoch: 30	 Loss: 0.5245	 Accuracy: 0.9356
# 100%|██████████| 1500/1500 [10:02<00:00,  2.49it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 31	 Loss: 0.5232	 Accuracy: 0.9357
# 100%|██████████| 1500/1500 [12:56<00:00,  1.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 32	 Loss: 0.5223	 Accuracy: 0.9367
# 100%|██████████| 1500/1500 [13:30<00:00,  1.85it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 33	 Loss: 0.5213	 Accuracy: 0.9368
# 100%|██████████| 1500/1500 [13:51<00:00,  1.80it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 34	 Loss: 0.5197	 Accuracy: 0.9379
# 100%|██████████| 1500/1500 [12:29<00:00,  2.00it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 35	 Loss: 0.5186	 Accuracy: 0.9368
# 100%|██████████| 1500/1500 [12:04<00:00,  2.07it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 36	 Loss: 0.5176	 Accuracy: 0.9377
# 100%|██████████| 1500/1500 [12:20<00:00,  2.03it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 37	 Loss: 0.5167	 Accuracy: 0.9374
# 100%|██████████| 1500/1500 [11:31<00:00,  2.17it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 38	 Loss: 0.5158	 Accuracy: 0.9366
# 100%|██████████| 1500/1500 [09:29<00:00,  2.63it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 39	 Loss: 0.5150	 Accuracy: 0.9384
# 100%|██████████| 1500/1500 [09:21<00:00,  2.67it/s]
# Model: Kmeans 	 Epoch: 40	 Loss: 0.5145	 Accuracy: 0.9379
# 100%|██████████| 1500/1500 [08:57<00:00,  2.79it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 41	 Loss: 0.5134	 Accuracy: 0.9372
# 100%|██████████| 1500/1500 [09:11<00:00,  2.72it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 42	 Loss: 0.5127	 Accuracy: 0.9384
# 100%|██████████| 1500/1500 [08:54<00:00,  2.81it/s]
# Model: Kmeans 	 Epoch: 43	 Loss: 0.5120	 Accuracy: 0.9357
# 100%|██████████| 1500/1500 [12:46<00:00,  1.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 44	 Loss: 0.5113	 Accuracy: 0.9382
# 100%|██████████| 1500/1500 [10:14<00:00,  2.44it/s]
# Model: Kmeans 	 Epoch: 45	 Loss: 0.5109	 Accuracy: 0.9384
# 100%|██████████| 1500/1500 [09:26<00:00,  2.65it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 46	 Loss: 0.5096	 Accuracy: 0.9391
# 100%|██████████| 1500/1500 [09:18<00:00,  2.69it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 47	 Loss: 0.5095	 Accuracy: 0.9379
# 100%|██████████| 1500/1500 [09:19<00:00,  2.68it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 48	 Loss: 0.5085	 Accuracy: 0.9392
# 100%|██████████| 1500/1500 [09:52<00:00,  2.53it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 49	 Loss: 0.5078	 Accuracy: 0.9391
# 100%|██████████| 1500/1500 [12:14<00:00,  2.04it/s]
# Model: Kmeans 	 Epoch: 50	 Loss: 0.5074	 Accuracy: 0.9387
# Save plot as ./alpha_value/alpha_value_0.8.png
# Model 1 name: TeacherModelCNN
# Model: TeacherModel	 Epoch: 1	 Loss: 0.4954	 Accuracy: 0.9271
# Model: TeacherModel	 Epoch: 2	 Loss: 0.2619	 Accuracy: 0.9433
# Model: TeacherModel	 Epoch: 3	 Loss: 0.2088	 Accuracy: 0.9548
# Model: TeacherModel	 Epoch: 4	 Loss: 0.1813	 Accuracy: 0.9606
# Model: TeacherModel	 Epoch: 5	 Loss: 0.1619	 Accuracy: 0.9537
# Model: TeacherModel	 Epoch: 6	 Loss: 0.1501	 Accuracy: 0.9676
# Model: TeacherModel	 Epoch: 7	 Loss: 0.1371	 Accuracy: 0.9672
# Model: TeacherModel	 Epoch: 8	 Loss: 0.1277	 Accuracy: 0.9709
# Model: TeacherModel	 Epoch: 9	 Loss: 0.1229	 Accuracy: 0.9651
# Model: TeacherModel	 Epoch: 10	 Loss: 0.1149	 Accuracy: 0.9728
# Model: TeacherModel	 Epoch: 11	 Loss: 0.1097	 Accuracy: 0.9710
# Model: TeacherModel	 Epoch: 12	 Loss: 0.1012	 Accuracy: 0.9742
# Model: TeacherModel	 Epoch: 13	 Loss: 0.0997	 Accuracy: 0.9744
# Model: TeacherModel	 Epoch: 14	 Loss: 0.0957	 Accuracy: 0.9747
# Model: TeacherModel	 Epoch: 15	 Loss: 0.0944	 Accuracy: 0.9758
# Model: TeacherModel	 Epoch: 16	 Loss: 0.0923	 Accuracy: 0.9752
# Model: TeacherModel	 Epoch: 17	 Loss: 0.0870	 Accuracy: 0.9759
# Model: TeacherModel	 Epoch: 18	 Loss: 0.0834	 Accuracy: 0.9784
# Model: TeacherModel	 Epoch: 19	 Loss: 0.0812	 Accuracy: 0.9762
# Model: TeacherModel	 Epoch: 20	 Loss: 0.0783	 Accuracy: 0.9757
# Model: TeacherModel	 Epoch: 21	 Loss: 0.0762	 Accuracy: 0.9767
# Model: TeacherModel	 Epoch: 22	 Loss: 0.0739	 Accuracy: 0.9788
# Model: TeacherModel	 Epoch: 23	 Loss: 0.0750	 Accuracy: 0.9800
# Model: TeacherModel	 Epoch: 24	 Loss: 0.0691	 Accuracy: 0.9749
# Model: TeacherModel	 Epoch: 25	 Loss: 0.0698	 Accuracy: 0.9769
# Model: TeacherModel	 Epoch: 26	 Loss: 0.0701	 Accuracy: 0.9786
# Model: TeacherModel	 Epoch: 27	 Loss: 0.0681	 Accuracy: 0.9805
# Model: TeacherModel	 Epoch: 28	 Loss: 0.0653	 Accuracy: 0.9801
# Model: TeacherModel	 Epoch: 29	 Loss: 0.0616	 Accuracy: 0.9789
# Model: TeacherModel	 Epoch: 30	 Loss: 0.0629	 Accuracy: 0.9798
# Model: TeacherModel	 Epoch: 31	 Loss: 0.0632	 Accuracy: 0.9776
# Model: TeacherModel	 Epoch: 32	 Loss: 0.0588	 Accuracy: 0.9793
# Model: TeacherModel	 Epoch: 33	 Loss: 0.0617	 Accuracy: 0.9822
# Model: TeacherModel	 Epoch: 34	 Loss: 0.0591	 Accuracy: 0.9799
# Model: TeacherModel	 Epoch: 35	 Loss: 0.0580	 Accuracy: 0.9810
# Model: TeacherModel	 Epoch: 36	 Loss: 0.0562	 Accuracy: 0.9798
# Model: TeacherModel	 Epoch: 37	 Loss: 0.0561	 Accuracy: 0.9804
# Model: TeacherModel	 Epoch: 38	 Loss: 0.0551	 Accuracy: 0.9803
# Model: TeacherModel	 Epoch: 39	 Loss: 0.0531	 Accuracy: 0.9802
# Model: TeacherModel	 Epoch: 40	 Loss: 0.0536	 Accuracy: 0.9798
# Model: TeacherModel	 Epoch: 41	 Loss: 0.0512	 Accuracy: 0.9821
# Model: TeacherModel	 Epoch: 42	 Loss: 0.0537	 Accuracy: 0.9790
# Model: TeacherModel	 Epoch: 43	 Loss: 0.0485	 Accuracy: 0.9828
# Model: TeacherModel	 Epoch: 44	 Loss: 0.0501	 Accuracy: 0.9794
# Model: TeacherModel	 Epoch: 45	 Loss: 0.0502	 Accuracy: 0.9822
# Model: TeacherModel	 Epoch: 46	 Loss: 0.0492	 Accuracy: 0.9828
# Model: TeacherModel	 Epoch: 47	 Loss: 0.0500	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 48	 Loss: 0.0477	 Accuracy: 0.9801
# Model: TeacherModel	 Epoch: 49	 Loss: 0.0476	 Accuracy: 0.9830
# Model: TeacherModel	 Epoch: 50	 Loss: 0.0470	 Accuracy: 0.9824
# Model 2 name: StudentModelA
# Model: StudentModel	 Epoch: 1	 Loss: 0.9184	 Accuracy: 0.8652
# Model: StudentModel	 Epoch: 2	 Loss: 0.3924	 Accuracy: 0.8937
# Model: StudentModel	 Epoch: 3	 Loss: 0.3354	 Accuracy: 0.9042
# Model: StudentModel	 Epoch: 4	 Loss: 0.3129	 Accuracy: 0.9057
# Model: StudentModel	 Epoch: 5	 Loss: 0.2976	 Accuracy: 0.9116
# Model: StudentModel	 Epoch: 6	 Loss: 0.2869	 Accuracy: 0.9125
# Model: StudentModel	 Epoch: 7	 Loss: 0.2774	 Accuracy: 0.9096
# Model: StudentModel	 Epoch: 8	 Loss: 0.2700	 Accuracy: 0.9169
# Model: StudentModel	 Epoch: 9	 Loss: 0.2623	 Accuracy: 0.9177
# Model: StudentModel	 Epoch: 10	 Loss: 0.2563	 Accuracy: 0.9203
# Model: StudentModel	 Epoch: 11	 Loss: 0.2511	 Accuracy: 0.9211
# Model: StudentModel	 Epoch: 12	 Loss: 0.2459	 Accuracy: 0.9225
# Model: StudentModel	 Epoch: 13	 Loss: 0.2404	 Accuracy: 0.9223
# Model: StudentModel	 Epoch: 14	 Loss: 0.2357	 Accuracy: 0.9213
# Model: StudentModel	 Epoch: 15	 Loss: 0.2301	 Accuracy: 0.9256
# Model: StudentModel	 Epoch: 16	 Loss: 0.2256	 Accuracy: 0.9260
# Model: StudentModel	 Epoch: 17	 Loss: 0.2203	 Accuracy: 0.9266
# Model: StudentModel	 Epoch: 18	 Loss: 0.2157	 Accuracy: 0.9252
# Model: StudentModel	 Epoch: 19	 Loss: 0.2120	 Accuracy: 0.9298
# Model: StudentModel	 Epoch: 20	 Loss: 0.2080	 Accuracy: 0.9305
# Model: StudentModel	 Epoch: 21	 Loss: 0.2034	 Accuracy: 0.9306
# Model: StudentModel	 Epoch: 22	 Loss: 0.2000	 Accuracy: 0.9293
# Model: StudentModel	 Epoch: 23	 Loss: 0.1970	 Accuracy: 0.9309
# Model: StudentModel	 Epoch: 24	 Loss: 0.1934	 Accuracy: 0.9338
# Model: StudentModel	 Epoch: 25	 Loss: 0.1912	 Accuracy: 0.9341
# Model: StudentModel	 Epoch: 26	 Loss: 0.1890	 Accuracy: 0.9359
# Model: StudentModel	 Epoch: 27	 Loss: 0.1853	 Accuracy: 0.9364
# Model: StudentModel	 Epoch: 28	 Loss: 0.1842	 Accuracy: 0.9363
# Model: StudentModel	 Epoch: 29	 Loss: 0.1817	 Accuracy: 0.9303
# Model: StudentModel	 Epoch: 30	 Loss: 0.1786	 Accuracy: 0.9370
# Model: StudentModel	 Epoch: 31	 Loss: 0.1776	 Accuracy: 0.9387
# Model: StudentModel	 Epoch: 32	 Loss: 0.1757	 Accuracy: 0.9363
# Model: StudentModel	 Epoch: 33	 Loss: 0.1732	 Accuracy: 0.9387
# Model: StudentModel	 Epoch: 34	 Loss: 0.1721	 Accuracy: 0.9387
# Model: StudentModel	 Epoch: 35	 Loss: 0.1705	 Accuracy: 0.9400
# Model: StudentModel	 Epoch: 36	 Loss: 0.1693	 Accuracy: 0.9381
# Model: StudentModel	 Epoch: 37	 Loss: 0.1676	 Accuracy: 0.9415
# Model: StudentModel	 Epoch: 38	 Loss: 0.1657	 Accuracy: 0.9406
# Model: StudentModel	 Epoch: 39	 Loss: 0.1642	 Accuracy: 0.9397
# Model: StudentModel	 Epoch: 40	 Loss: 0.1627	 Accuracy: 0.9421
# Model: StudentModel	 Epoch: 41	 Loss: 0.1611	 Accuracy: 0.9353
# Model: StudentModel	 Epoch: 42	 Loss: 0.1597	 Accuracy: 0.9410
# Model: StudentModel	 Epoch: 43	 Loss: 0.1595	 Accuracy: 0.9427
# Model: StudentModel	 Epoch: 44	 Loss: 0.1577	 Accuracy: 0.9413
# Model: StudentModel	 Epoch: 45	 Loss: 0.1560	 Accuracy: 0.9416
# Model: StudentModel	 Epoch: 46	 Loss: 0.1558	 Accuracy: 0.9403
# Model: StudentModel	 Epoch: 47	 Loss: 0.1542	 Accuracy: 0.9409
# Model: StudentModel	 Epoch: 48	 Loss: 0.1525	 Accuracy: 0.9392
# Model: StudentModel	 Epoch: 49	 Loss: 0.1519	 Accuracy: 0.9443
# Model: StudentModel	 Epoch: 50	 Loss: 0.1500	 Accuracy: 0.9427
# Model 3 name: StudentModelB
# Model: StudentModel	 Epoch: 1	 Loss: 26.2706	 Accuracy: 0.7962
# Model: StudentModel	 Epoch: 2	 Loss: 21.4888	 Accuracy: 0.8636
# Model: StudentModel	 Epoch: 3	 Loss: 20.6076	 Accuracy: 0.8840
# Model: StudentModel	 Epoch: 4	 Loss: 20.2699	 Accuracy: 0.8870
# Model: StudentModel	 Epoch: 5	 Loss: 20.1025	 Accuracy: 0.8955
# Model: StudentModel	 Epoch: 6	 Loss: 19.9970	 Accuracy: 0.8993
# Model: StudentModel	 Epoch: 7	 Loss: 19.9184	 Accuracy: 0.8998
# Model: StudentModel	 Epoch: 8	 Loss: 19.8572	 Accuracy: 0.9026
# Model: StudentModel	 Epoch: 9	 Loss: 19.8025	 Accuracy: 0.9049
# Model: StudentModel	 Epoch: 10	 Loss: 19.7580	 Accuracy: 0.9074
# Model: StudentModel	 Epoch: 11	 Loss: 19.7142	 Accuracy: 0.9076
# Model: StudentModel	 Epoch: 12	 Loss: 19.6766	 Accuracy: 0.9076
# Model: StudentModel	 Epoch: 13	 Loss: 19.6451	 Accuracy: 0.9089
# Model: StudentModel	 Epoch: 14	 Loss: 19.6135	 Accuracy: 0.9101
# Model: StudentModel	 Epoch: 15	 Loss: 19.5851	 Accuracy: 0.9128
# Model: StudentModel	 Epoch: 16	 Loss: 19.5612	 Accuracy: 0.9107
# Model: StudentModel	 Epoch: 17	 Loss: 19.5301	 Accuracy: 0.9133
# Model: StudentModel	 Epoch: 18	 Loss: 19.5085	 Accuracy: 0.9127
# Model: StudentModel	 Epoch: 19	 Loss: 19.4809	 Accuracy: 0.9143
# Model: StudentModel	 Epoch: 20	 Loss: 19.4548	 Accuracy: 0.9139
# Model: StudentModel	 Epoch: 21	 Loss: 19.4316	 Accuracy: 0.9144
# Model: StudentModel	 Epoch: 22	 Loss: 19.4124	 Accuracy: 0.9159
# Model: StudentModel	 Epoch: 23	 Loss: 19.3898	 Accuracy: 0.9133
# Model: StudentModel	 Epoch: 24	 Loss: 19.3693	 Accuracy: 0.9177
# Model: StudentModel	 Epoch: 25	 Loss: 19.3481	 Accuracy: 0.9180
# Model: StudentModel	 Epoch: 26	 Loss: 19.3358	 Accuracy: 0.9188
# Model: StudentModel	 Epoch: 27	 Loss: 19.3151	 Accuracy: 0.9185
# Model: StudentModel	 Epoch: 28	 Loss: 19.2983	 Accuracy: 0.9178
# Model: StudentModel	 Epoch: 29	 Loss: 19.2831	 Accuracy: 0.9203
# Model: StudentModel	 Epoch: 30	 Loss: 19.2752	 Accuracy: 0.9227
# Model: StudentModel	 Epoch: 31	 Loss: 19.2561	 Accuracy: 0.9201
# Model: StudentModel	 Epoch: 32	 Loss: 19.2418	 Accuracy: 0.9215
# Model: StudentModel	 Epoch: 33	 Loss: 19.2299	 Accuracy: 0.9206
# Model: StudentModel	 Epoch: 34	 Loss: 19.2177	 Accuracy: 0.9194
# Model: StudentModel	 Epoch: 35	 Loss: 19.2029	 Accuracy: 0.9168
# Model: StudentModel	 Epoch: 36	 Loss: 19.1912	 Accuracy: 0.9185
# Model: StudentModel	 Epoch: 37	 Loss: 19.1809	 Accuracy: 0.9239
# Model: StudentModel	 Epoch: 38	 Loss: 19.1679	 Accuracy: 0.9230
# Model: StudentModel	 Epoch: 39	 Loss: 19.1558	 Accuracy: 0.9214
# Model: StudentModel	 Epoch: 40	 Loss: 19.1486	 Accuracy: 0.9251
# Model: StudentModel	 Epoch: 41	 Loss: 19.1337	 Accuracy: 0.9222
# Model: StudentModel	 Epoch: 42	 Loss: 19.1231	 Accuracy: 0.9250
# Model: StudentModel	 Epoch: 43	 Loss: 19.1114	 Accuracy: 0.9268
# Model: StudentModel	 Epoch: 44	 Loss: 19.0986	 Accuracy: 0.9243
# Model: StudentModel	 Epoch: 45	 Loss: 19.0913	 Accuracy: 0.9258
# Model: StudentModel	 Epoch: 46	 Loss: 19.0816	 Accuracy: 0.9247
# Model: StudentModel	 Epoch: 47	 Loss: 19.0706	 Accuracy: 0.9238
# Model: StudentModel	 Epoch: 48	 Loss: 19.0607	 Accuracy: 0.9273
# Model: StudentModel	 Epoch: 49	 Loss: 19.0497	 Accuracy: 0.9260
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: StudentModel	 Epoch: 50	 Loss: 19.0460	 Accuracy: 0.9263
# Model 4 name: StudentModelC
# 100%|██████████| 1500/1500 [08:33<00:00,  2.92it/s]
# Model: Kmeans 	 Epoch: 1	 Loss: 0.9569	 Accuracy: 0.8641
# 100%|██████████| 1500/1500 [08:36<00:00,  2.90it/s]
# Model: Kmeans 	 Epoch: 2	 Loss: 0.7565	 Accuracy: 0.8827
# 100%|██████████| 1500/1500 [08:33<00:00,  2.92it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 3	 Loss: 0.7108	 Accuracy: 0.8946
# 100%|██████████| 1500/1500 [09:53<00:00,  2.53it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 4	 Loss: 0.6850	 Accuracy: 0.9021
# 100%|██████████| 1500/1500 [12:23<00:00,  2.02it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 5	 Loss: 0.6686	 Accuracy: 0.9067
# 100%|██████████| 1500/1500 [09:58<00:00,  2.51it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 6	 Loss: 0.6570	 Accuracy: 0.9103
# 100%|██████████| 1500/1500 [08:36<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 7	 Loss: 0.6482	 Accuracy: 0.9086
# 100%|██████████| 1500/1500 [08:35<00:00,  2.91it/s]
# Model: Kmeans 	 Epoch: 8	 Loss: 0.6406	 Accuracy: 0.9134
# 100%|██████████| 1500/1500 [08:36<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 9	 Loss: 0.6349	 Accuracy: 0.9136
# 100%|██████████| 1500/1500 [08:38<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 10	 Loss: 0.6292	 Accuracy: 0.9130
# 100%|██████████| 1500/1500 [08:39<00:00,  2.88it/s]
# Model: Kmeans 	 Epoch: 11	 Loss: 0.6243	 Accuracy: 0.9188
# 100%|██████████| 1500/1500 [08:39<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 12	 Loss: 0.6199	 Accuracy: 0.9197
# 100%|██████████| 1500/1500 [08:36<00:00,  2.91it/s]
# Model: Kmeans 	 Epoch: 13	 Loss: 0.6163	 Accuracy: 0.9201
# 100%|██████████| 1500/1500 [08:38<00:00,  2.89it/s]
# Model: Kmeans 	 Epoch: 14	 Loss: 0.6130	 Accuracy: 0.9183
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 15	 Loss: 0.6099	 Accuracy: 0.9206
# 100%|██████████| 1500/1500 [08:35<00:00,  2.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 16	 Loss: 0.6072	 Accuracy: 0.9200
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
# Model: Kmeans 	 Epoch: 17	 Loss: 0.6048	 Accuracy: 0.9228
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 18	 Loss: 0.6019	 Accuracy: 0.9233
# 100%|██████████| 1500/1500 [08:38<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 19	 Loss: 0.5995	 Accuracy: 0.9217
# 100%|██████████| 1500/1500 [08:36<00:00,  2.90it/s]
# Model: Kmeans 	 Epoch: 20	 Loss: 0.5971	 Accuracy: 0.9256
# 100%|██████████| 1500/1500 [08:36<00:00,  2.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 21	 Loss: 0.5948	 Accuracy: 0.9230
# 100%|██████████| 1500/1500 [08:34<00:00,  2.91it/s]
# Model: Kmeans 	 Epoch: 22	 Loss: 0.5929	 Accuracy: 0.9227
# 100%|██████████| 1500/1500 [08:35<00:00,  2.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 23	 Loss: 0.5909	 Accuracy: 0.9253
# 100%|██████████| 1500/1500 [08:35<00:00,  2.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 24	 Loss: 0.5893	 Accuracy: 0.9224
# 100%|██████████| 1500/1500 [08:35<00:00,  2.91it/s]
# Model: Kmeans 	 Epoch: 25	 Loss: 0.5873	 Accuracy: 0.9230
# 100%|██████████| 1500/1500 [08:34<00:00,  2.92it/s]
# Model: Kmeans 	 Epoch: 26	 Loss: 0.5854	 Accuracy: 0.9224
# 100%|██████████| 1500/1500 [08:32<00:00,  2.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 27	 Loss: 0.5839	 Accuracy: 0.9254
# 100%|██████████| 1500/1500 [08:30<00:00,  2.94it/s]
# Model: Kmeans 	 Epoch: 28	 Loss: 0.5818	 Accuracy: 0.9252
# 100%|██████████| 1500/1500 [08:30<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 29	 Loss: 0.5808	 Accuracy: 0.9277
# 100%|██████████| 1500/1500 [08:30<00:00,  2.94it/s]
# Model: Kmeans 	 Epoch: 30	 Loss: 0.5793	 Accuracy: 0.9283
# 100%|██████████| 1500/1500 [08:29<00:00,  2.94it/s]
# Model: Kmeans 	 Epoch: 31	 Loss: 0.5777	 Accuracy: 0.9236
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 32	 Loss: 0.5771	 Accuracy: 0.9280
# 100%|██████████| 1500/1500 [08:29<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 33	 Loss: 0.5757	 Accuracy: 0.9286
# 100%|██████████| 1500/1500 [08:28<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 34	 Loss: 0.5749	 Accuracy: 0.9295
# 100%|██████████| 1500/1500 [08:34<00:00,  2.92it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 35	 Loss: 0.5738	 Accuracy: 0.9288
# 100%|██████████| 1500/1500 [08:30<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 36	 Loss: 0.5724	 Accuracy: 0.9289
# 100%|██████████| 1500/1500 [08:29<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 37	 Loss: 0.5725	 Accuracy: 0.9256
# 100%|██████████| 1500/1500 [08:29<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 38	 Loss: 0.5712	 Accuracy: 0.9309
# 100%|██████████| 1500/1500 [08:33<00:00,  2.92it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 39	 Loss: 0.5703	 Accuracy: 0.9258
# 100%|██████████| 1500/1500 [08:30<00:00,  2.94it/s]
# Model: Kmeans 	 Epoch: 40	 Loss: 0.5692	 Accuracy: 0.9305
# 100%|██████████| 1500/1500 [08:30<00:00,  2.94it/s]
# Model: Kmeans 	 Epoch: 41	 Loss: 0.5685	 Accuracy: 0.9317
# 100%|██████████| 1500/1500 [08:28<00:00,  2.95it/s]
# Model: Kmeans 	 Epoch: 42	 Loss: 0.5677	 Accuracy: 0.9306
# 100%|██████████| 1500/1500 [08:32<00:00,  2.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 43	 Loss: 0.5675	 Accuracy: 0.9323
# 100%|██████████| 1500/1500 [08:32<00:00,  2.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 44	 Loss: 0.5667	 Accuracy: 0.9287
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 45	 Loss: 0.5659	 Accuracy: 0.9293
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
# Model: Kmeans 	 Epoch: 46	 Loss: 0.5653	 Accuracy: 0.9307
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 47	 Loss: 0.5648	 Accuracy: 0.9278
# 100%|██████████| 1500/1500 [08:30<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 48	 Loss: 0.5643	 Accuracy: 0.9290
# 100%|██████████| 1500/1500 [08:32<00:00,  2.93it/s]
# Model: Kmeans 	 Epoch: 49	 Loss: 0.5637	 Accuracy: 0.9313
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
# Model: Kmeans 	 Epoch: 50	 Loss: 0.5630	 Accuracy: 0.9271
# Save plot as ./alpha_value/alpha_value_0.7.png
# Model 1 name: TeacherModelCNN
# Model: TeacherModel	 Epoch: 1	 Loss: 0.4902	 Accuracy: 0.9296
# Model: TeacherModel	 Epoch: 2	 Loss: 0.2571	 Accuracy: 0.9455
# Model: TeacherModel	 Epoch: 3	 Loss: 0.2049	 Accuracy: 0.9525
# Model: TeacherModel	 Epoch: 4	 Loss: 0.1762	 Accuracy: 0.9576
# Model: TeacherModel	 Epoch: 5	 Loss: 0.1603	 Accuracy: 0.9595
# Model: TeacherModel	 Epoch: 6	 Loss: 0.1521	 Accuracy: 0.9663
# Model: TeacherModel	 Epoch: 7	 Loss: 0.1364	 Accuracy: 0.9660
# Model: TeacherModel	 Epoch: 8	 Loss: 0.1285	 Accuracy: 0.9685
# Model: TeacherModel	 Epoch: 9	 Loss: 0.1197	 Accuracy: 0.9713
# Model: TeacherModel	 Epoch: 10	 Loss: 0.1133	 Accuracy: 0.9743
# Model: TeacherModel	 Epoch: 11	 Loss: 0.1052	 Accuracy: 0.9715
# Model: TeacherModel	 Epoch: 12	 Loss: 0.1041	 Accuracy: 0.9728
# Model: TeacherModel	 Epoch: 13	 Loss: 0.0959	 Accuracy: 0.9749
# Model: TeacherModel	 Epoch: 14	 Loss: 0.0972	 Accuracy: 0.9769
# Model: TeacherModel	 Epoch: 15	 Loss: 0.0931	 Accuracy: 0.9758
# Model: TeacherModel	 Epoch: 16	 Loss: 0.0903	 Accuracy: 0.9750
# Model: TeacherModel	 Epoch: 17	 Loss: 0.0873	 Accuracy: 0.9756
# Model: TeacherModel	 Epoch: 18	 Loss: 0.0845	 Accuracy: 0.9798
# Model: TeacherModel	 Epoch: 19	 Loss: 0.0820	 Accuracy: 0.9771
# Model: TeacherModel	 Epoch: 20	 Loss: 0.0805	 Accuracy: 0.9802
# Model: TeacherModel	 Epoch: 21	 Loss: 0.0793	 Accuracy: 0.9745
# Model: TeacherModel	 Epoch: 22	 Loss: 0.0724	 Accuracy: 0.9806
# Model: TeacherModel	 Epoch: 23	 Loss: 0.0711	 Accuracy: 0.9778
# Model: TeacherModel	 Epoch: 24	 Loss: 0.0712	 Accuracy: 0.9778
# Model: TeacherModel	 Epoch: 25	 Loss: 0.0727	 Accuracy: 0.9797
# Model: TeacherModel	 Epoch: 26	 Loss: 0.0664	 Accuracy: 0.9788
# Model: TeacherModel	 Epoch: 27	 Loss: 0.0673	 Accuracy: 0.9805
# Model: TeacherModel	 Epoch: 28	 Loss: 0.0671	 Accuracy: 0.9785
# Model: TeacherModel	 Epoch: 29	 Loss: 0.0635	 Accuracy: 0.9793
# Model: TeacherModel	 Epoch: 30	 Loss: 0.0613	 Accuracy: 0.9807
# Model: TeacherModel	 Epoch: 31	 Loss: 0.0621	 Accuracy: 0.9812
# Model: TeacherModel	 Epoch: 32	 Loss: 0.0598	 Accuracy: 0.9805
# Model: TeacherModel	 Epoch: 33	 Loss: 0.0584	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 34	 Loss: 0.0592	 Accuracy: 0.9800
# Model: TeacherModel	 Epoch: 35	 Loss: 0.0580	 Accuracy: 0.9812
# Model: TeacherModel	 Epoch: 36	 Loss: 0.0576	 Accuracy: 0.9801
# Model: TeacherModel	 Epoch: 37	 Loss: 0.0558	 Accuracy: 0.9797
# Model: TeacherModel	 Epoch: 38	 Loss: 0.0554	 Accuracy: 0.9812
# Model: TeacherModel	 Epoch: 39	 Loss: 0.0539	 Accuracy: 0.9802
# Model: TeacherModel	 Epoch: 40	 Loss: 0.0551	 Accuracy: 0.9818
# Model: TeacherModel	 Epoch: 41	 Loss: 0.0519	 Accuracy: 0.9804
# Model: TeacherModel	 Epoch: 42	 Loss: 0.0490	 Accuracy: 0.9811
# Model: TeacherModel	 Epoch: 43	 Loss: 0.0526	 Accuracy: 0.9814
# Model: TeacherModel	 Epoch: 44	 Loss: 0.0499	 Accuracy: 0.9814
# Model: TeacherModel	 Epoch: 45	 Loss: 0.0495	 Accuracy: 0.9818
# Model: TeacherModel	 Epoch: 46	 Loss: 0.0486	 Accuracy: 0.9797
# Model: TeacherModel	 Epoch: 47	 Loss: 0.0486	 Accuracy: 0.9825
# Model: TeacherModel	 Epoch: 48	 Loss: 0.0516	 Accuracy: 0.9812
# Model: TeacherModel	 Epoch: 49	 Loss: 0.0491	 Accuracy: 0.9826
# Model: TeacherModel	 Epoch: 50	 Loss: 0.0455	 Accuracy: 0.9829
# Model 2 name: StudentModelA
# Model: StudentModel	 Epoch: 1	 Loss: 0.7685	 Accuracy: 0.8808
# Model: StudentModel	 Epoch: 2	 Loss: 0.3643	 Accuracy: 0.9007
# Model: StudentModel	 Epoch: 3	 Loss: 0.3175	 Accuracy: 0.9086
# Model: StudentModel	 Epoch: 4	 Loss: 0.2940	 Accuracy: 0.9141
# Model: StudentModel	 Epoch: 5	 Loss: 0.2782	 Accuracy: 0.9150
# Model: StudentModel	 Epoch: 6	 Loss: 0.2662	 Accuracy: 0.9167
# Model: StudentModel	 Epoch: 7	 Loss: 0.2558	 Accuracy: 0.9230
# Model: StudentModel	 Epoch: 8	 Loss: 0.2454	 Accuracy: 0.9215
# Model: StudentModel	 Epoch: 9	 Loss: 0.2388	 Accuracy: 0.9256
# Model: StudentModel	 Epoch: 10	 Loss: 0.2296	 Accuracy: 0.9220
# Model: StudentModel	 Epoch: 11	 Loss: 0.2230	 Accuracy: 0.9273
# Model: StudentModel	 Epoch: 12	 Loss: 0.2174	 Accuracy: 0.9294
# Model: StudentModel	 Epoch: 13	 Loss: 0.2115	 Accuracy: 0.9311
# Model: StudentModel	 Epoch: 14	 Loss: 0.2050	 Accuracy: 0.9340
# Model: StudentModel	 Epoch: 15	 Loss: 0.2005	 Accuracy: 0.9296
# Model: StudentModel	 Epoch: 16	 Loss: 0.1953	 Accuracy: 0.9367
# Model: StudentModel	 Epoch: 17	 Loss: 0.1910	 Accuracy: 0.9374
# Model: StudentModel	 Epoch: 18	 Loss: 0.1869	 Accuracy: 0.9364
# Model: StudentModel	 Epoch: 19	 Loss: 0.1837	 Accuracy: 0.9372
# Model: StudentModel	 Epoch: 20	 Loss: 0.1797	 Accuracy: 0.9371
# Model: StudentModel	 Epoch: 21	 Loss: 0.1768	 Accuracy: 0.9390
# Model: StudentModel	 Epoch: 22	 Loss: 0.1738	 Accuracy: 0.9397
# Model: StudentModel	 Epoch: 23	 Loss: 0.1705	 Accuracy: 0.9425
# Model: StudentModel	 Epoch: 24	 Loss: 0.1679	 Accuracy: 0.9399
# Model: StudentModel	 Epoch: 25	 Loss: 0.1659	 Accuracy: 0.9429
# Model: StudentModel	 Epoch: 26	 Loss: 0.1627	 Accuracy: 0.9423
# Model: StudentModel	 Epoch: 27	 Loss: 0.1599	 Accuracy: 0.9455
# Model: StudentModel	 Epoch: 28	 Loss: 0.1579	 Accuracy: 0.9420
# Model: StudentModel	 Epoch: 29	 Loss: 0.1562	 Accuracy: 0.9413
# Model: StudentModel	 Epoch: 30	 Loss: 0.1549	 Accuracy: 0.9452
# Model: StudentModel	 Epoch: 31	 Loss: 0.1526	 Accuracy: 0.9443
# Model: StudentModel	 Epoch: 32	 Loss: 0.1508	 Accuracy: 0.9455
# Model: StudentModel	 Epoch: 33	 Loss: 0.1495	 Accuracy: 0.9472
# Model: StudentModel	 Epoch: 34	 Loss: 0.1475	 Accuracy: 0.9473
# Model: StudentModel	 Epoch: 35	 Loss: 0.1469	 Accuracy: 0.9447
# Model: StudentModel	 Epoch: 36	 Loss: 0.1457	 Accuracy: 0.9445
# Model: StudentModel	 Epoch: 37	 Loss: 0.1440	 Accuracy: 0.9470
# Model: StudentModel	 Epoch: 38	 Loss: 0.1420	 Accuracy: 0.9450
# Model: StudentModel	 Epoch: 39	 Loss: 0.1413	 Accuracy: 0.9461
# Model: StudentModel	 Epoch: 40	 Loss: 0.1401	 Accuracy: 0.9477
# Model: StudentModel	 Epoch: 41	 Loss: 0.1389	 Accuracy: 0.9434
# Model: StudentModel	 Epoch: 42	 Loss: 0.1383	 Accuracy: 0.9469
# Model: StudentModel	 Epoch: 43	 Loss: 0.1368	 Accuracy: 0.9448
# Model: StudentModel	 Epoch: 44	 Loss: 0.1359	 Accuracy: 0.9466
# Model: StudentModel	 Epoch: 45	 Loss: 0.1353	 Accuracy: 0.9467
# Model: StudentModel	 Epoch: 46	 Loss: 0.1337	 Accuracy: 0.9488
# Model: StudentModel	 Epoch: 47	 Loss: 0.1321	 Accuracy: 0.9452
# Model: StudentModel	 Epoch: 48	 Loss: 0.1324	 Accuracy: 0.9469
# Model: StudentModel	 Epoch: 49	 Loss: 0.1312	 Accuracy: 0.9458
# Model: StudentModel	 Epoch: 50	 Loss: 0.1307	 Accuracy: 0.9404
# Model 3 name: StudentModelB
# Model: StudentModel	 Epoch: 1	 Loss: 25.6733	 Accuracy: 0.8552
# Model: StudentModel	 Epoch: 2	 Loss: 20.7120	 Accuracy: 0.8815
# Model: StudentModel	 Epoch: 3	 Loss: 20.1869	 Accuracy: 0.8921
# Model: StudentModel	 Epoch: 4	 Loss: 19.9773	 Accuracy: 0.8988
# Model: StudentModel	 Epoch: 5	 Loss: 19.8572	 Accuracy: 0.9015
# Model: StudentModel	 Epoch: 6	 Loss: 19.7637	 Accuracy: 0.9046
# Model: StudentModel	 Epoch: 7	 Loss: 19.6910	 Accuracy: 0.9061
# Model: StudentModel	 Epoch: 8	 Loss: 19.6347	 Accuracy: 0.9084
# Model: StudentModel	 Epoch: 9	 Loss: 19.5777	 Accuracy: 0.9094
# Model: StudentModel	 Epoch: 10	 Loss: 19.5242	 Accuracy: 0.9110
# Model: StudentModel	 Epoch: 11	 Loss: 19.4743	 Accuracy: 0.9133
# Model: StudentModel	 Epoch: 12	 Loss: 19.4240	 Accuracy: 0.9162
# Model: StudentModel	 Epoch: 13	 Loss: 19.3769	 Accuracy: 0.9147
# Model: StudentModel	 Epoch: 14	 Loss: 19.3343	 Accuracy: 0.9194
# Model: StudentModel	 Epoch: 15	 Loss: 19.2910	 Accuracy: 0.9226
# Model: StudentModel	 Epoch: 16	 Loss: 19.2519	 Accuracy: 0.9223
# Model: StudentModel	 Epoch: 17	 Loss: 19.2167	 Accuracy: 0.9250
# Model: StudentModel	 Epoch: 18	 Loss: 19.1854	 Accuracy: 0.9247
# Model: StudentModel	 Epoch: 19	 Loss: 19.1573	 Accuracy: 0.9270
# Model: StudentModel	 Epoch: 20	 Loss: 19.1247	 Accuracy: 0.9283
# Model: StudentModel	 Epoch: 21	 Loss: 19.1008	 Accuracy: 0.9254
# Model: StudentModel	 Epoch: 22	 Loss: 19.0735	 Accuracy: 0.9280
# Model: StudentModel	 Epoch: 23	 Loss: 19.0512	 Accuracy: 0.9283
# Model: StudentModel	 Epoch: 24	 Loss: 19.0303	 Accuracy: 0.9304
# Model: StudentModel	 Epoch: 25	 Loss: 19.0090	 Accuracy: 0.9328
# Model: StudentModel	 Epoch: 26	 Loss: 18.9891	 Accuracy: 0.9313
# Model: StudentModel	 Epoch: 27	 Loss: 18.9729	 Accuracy: 0.9320
# Model: StudentModel	 Epoch: 28	 Loss: 18.9552	 Accuracy: 0.9332
# Model: StudentModel	 Epoch: 29	 Loss: 18.9400	 Accuracy: 0.9313
# Model: StudentModel	 Epoch: 30	 Loss: 18.9287	 Accuracy: 0.9331
# Model: StudentModel	 Epoch: 31	 Loss: 18.9152	 Accuracy: 0.9338
# Model: StudentModel	 Epoch: 32	 Loss: 18.8996	 Accuracy: 0.9351
# Model: StudentModel	 Epoch: 33	 Loss: 18.8883	 Accuracy: 0.9333
# Model: StudentModel	 Epoch: 34	 Loss: 18.8770	 Accuracy: 0.9363
# Model: StudentModel	 Epoch: 35	 Loss: 18.8686	 Accuracy: 0.9363
# Model: StudentModel	 Epoch: 36	 Loss: 18.8559	 Accuracy: 0.9366
# Model: StudentModel	 Epoch: 37	 Loss: 18.8495	 Accuracy: 0.9343
# Model: StudentModel	 Epoch: 38	 Loss: 18.8392	 Accuracy: 0.9379
# Model: StudentModel	 Epoch: 39	 Loss: 18.8317	 Accuracy: 0.9357
# Model: StudentModel	 Epoch: 40	 Loss: 18.8202	 Accuracy: 0.9378
# Model: StudentModel	 Epoch: 41	 Loss: 18.8123	 Accuracy: 0.9367
# Model: StudentModel	 Epoch: 42	 Loss: 18.8056	 Accuracy: 0.9395
# Model: StudentModel	 Epoch: 43	 Loss: 18.7982	 Accuracy: 0.9378
# Model: StudentModel	 Epoch: 44	 Loss: 18.7942	 Accuracy: 0.9396
# Model: StudentModel	 Epoch: 45	 Loss: 18.7822	 Accuracy: 0.9358
# Model: StudentModel	 Epoch: 46	 Loss: 18.7790	 Accuracy: 0.9396
# Model: StudentModel	 Epoch: 47	 Loss: 18.7725	 Accuracy: 0.9384
# Model: StudentModel	 Epoch: 48	 Loss: 18.7654	 Accuracy: 0.9364
# Model: StudentModel	 Epoch: 49	 Loss: 18.7605	 Accuracy: 0.9383
# Model: StudentModel	 Epoch: 50	 Loss: 18.7531	 Accuracy: 0.9373
# Model 4 name: StudentModelC
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 1	 Loss: 0.9025	 Accuracy: 0.8614
# 100%|██████████| 1500/1500 [08:27<00:00,  2.96it/s]
# Model: Kmeans 	 Epoch: 2	 Loss: 0.7293	 Accuracy: 0.8901
# 100%|██████████| 1500/1500 [08:29<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 3	 Loss: 0.6970	 Accuracy: 0.9012
# 100%|██████████| 1500/1500 [08:27<00:00,  2.96it/s]
# Model: Kmeans 	 Epoch: 4	 Loss: 0.6796	 Accuracy: 0.9030
# 100%|██████████| 1500/1500 [08:26<00:00,  2.96it/s]
# Model: Kmeans 	 Epoch: 5	 Loss: 0.6669	 Accuracy: 0.9042
# 100%|██████████| 1500/1500 [08:30<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 6	 Loss: 0.6573	 Accuracy: 0.9122
# 100%|██████████| 1500/1500 [08:29<00:00,  2.94it/s]
# Model: Kmeans 	 Epoch: 7	 Loss: 0.6504	 Accuracy: 0.9116
# 100%|██████████| 1500/1500 [08:29<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 8	 Loss: 0.6448	 Accuracy: 0.9147
# 100%|██████████| 1500/1500 [08:34<00:00,  2.92it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 9	 Loss: 0.6401	 Accuracy: 0.9173
# 100%|██████████| 1500/1500 [08:28<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 10	 Loss: 0.6355	 Accuracy: 0.9180
# 100%|██████████| 1500/1500 [08:28<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 11	 Loss: 0.6319	 Accuracy: 0.9193
# 100%|██████████| 1500/1500 [08:29<00:00,  2.94it/s]
# Model: Kmeans 	 Epoch: 12	 Loss: 0.6290	 Accuracy: 0.9206
# 100%|██████████| 1500/1500 [08:26<00:00,  2.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 13	 Loss: 0.6262	 Accuracy: 0.9196
# 100%|██████████| 1500/1500 [08:28<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 14	 Loss: 0.6238	 Accuracy: 0.9202
# 100%|██████████| 1500/1500 [08:29<00:00,  2.94it/s]
# Model: Kmeans 	 Epoch: 15	 Loss: 0.6212	 Accuracy: 0.9223
# 100%|██████████| 1500/1500 [08:27<00:00,  2.96it/s]
# Model: Kmeans 	 Epoch: 16	 Loss: 0.6189	 Accuracy: 0.9213
# 100%|██████████| 1500/1500 [08:26<00:00,  2.96it/s]
# Model: Kmeans 	 Epoch: 17	 Loss: 0.6167	 Accuracy: 0.9242
# 100%|██████████| 1500/1500 [08:29<00:00,  2.95it/s]
# Model: Kmeans 	 Epoch: 18	 Loss: 0.6143	 Accuracy: 0.9255
# 100%|██████████| 1500/1500 [08:28<00:00,  2.95it/s]
# Model: Kmeans 	 Epoch: 19	 Loss: 0.6124	 Accuracy: 0.9262
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
# Model: Kmeans 	 Epoch: 20	 Loss: 0.6103	 Accuracy: 0.9262
# 100%|██████████| 1500/1500 [08:28<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 21	 Loss: 0.6081	 Accuracy: 0.9242
# 100%|██████████| 1500/1500 [08:26<00:00,  2.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 22	 Loss: 0.6064	 Accuracy: 0.9249
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
# Model: Kmeans 	 Epoch: 23	 Loss: 0.6046	 Accuracy: 0.9263
# 100%|██████████| 1500/1500 [08:29<00:00,  2.94it/s]
# Model: Kmeans 	 Epoch: 24	 Loss: 0.6033	 Accuracy: 0.9280
# 100%|██████████| 1500/1500 [08:28<00:00,  2.95it/s]
# Model: Kmeans 	 Epoch: 25	 Loss: 0.6018	 Accuracy: 0.9273
# 100%|██████████| 1500/1500 [08:27<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 26	 Loss: 0.6004	 Accuracy: 0.9288
# 100%|██████████| 1500/1500 [08:27<00:00,  2.95it/s]
# Model: Kmeans 	 Epoch: 27	 Loss: 0.5989	 Accuracy: 0.9272
# 100%|██████████| 1500/1500 [08:27<00:00,  2.95it/s]
# Model: Kmeans 	 Epoch: 28	 Loss: 0.5978	 Accuracy: 0.9293
# 100%|██████████| 1500/1500 [08:27<00:00,  2.96it/s]
# Model: Kmeans 	 Epoch: 29	 Loss: 0.5968	 Accuracy: 0.9297
# 100%|██████████| 1500/1500 [08:29<00:00,  2.95it/s]
# Model: Kmeans 	 Epoch: 30	 Loss: 0.5954	 Accuracy: 0.9284
# 100%|██████████| 1500/1500 [08:26<00:00,  2.96it/s]
# Model: Kmeans 	 Epoch: 31	 Loss: 0.5946	 Accuracy: 0.9304
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
# Model: Kmeans 	 Epoch: 32	 Loss: 0.5934	 Accuracy: 0.9297
# 100%|██████████| 1500/1500 [08:29<00:00,  2.94it/s]
# Model: Kmeans 	 Epoch: 33	 Loss: 0.5925	 Accuracy: 0.9303
# 100%|██████████| 1500/1500 [08:27<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 34	 Loss: 0.5914	 Accuracy: 0.9313
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
# Model: Kmeans 	 Epoch: 35	 Loss: 0.5908	 Accuracy: 0.9316
# 100%|██████████| 1500/1500 [08:33<00:00,  2.92it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 36	 Loss: 0.5896	 Accuracy: 0.9323
# 100%|██████████| 1500/1500 [08:32<00:00,  2.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 37	 Loss: 0.5886	 Accuracy: 0.9299
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 38	 Loss: 0.5880	 Accuracy: 0.9315
# 100%|██████████| 1500/1500 [08:28<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 39	 Loss: 0.5870	 Accuracy: 0.9323
# 100%|██████████| 1500/1500 [08:29<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 40	 Loss: 0.5861	 Accuracy: 0.9346
# 100%|██████████| 1500/1500 [08:30<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 41	 Loss: 0.5857	 Accuracy: 0.9331
# 100%|██████████| 1500/1500 [08:28<00:00,  2.95it/s]
# Model: Kmeans 	 Epoch: 42	 Loss: 0.5850	 Accuracy: 0.9337
# 100%|██████████| 1500/1500 [08:28<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 43	 Loss: 0.5841	 Accuracy: 0.9345
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 44	 Loss: 0.5835	 Accuracy: 0.9333
# 100%|██████████| 1500/1500 [08:26<00:00,  2.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 45	 Loss: 0.5829	 Accuracy: 0.9319
# 100%|██████████| 1500/1500 [08:29<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 46	 Loss: 0.5821	 Accuracy: 0.9337
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 47	 Loss: 0.5816	 Accuracy: 0.9340
# 100%|██████████| 1500/1500 [08:30<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 48	 Loss: 0.5811	 Accuracy: 0.9353
# 100%|██████████| 1500/1500 [08:30<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 49	 Loss: 0.5805	 Accuracy: 0.9337
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
# Model: Kmeans 	 Epoch: 50	 Loss: 0.5801	 Accuracy: 0.9339
# Save plot as ./alpha_value/alpha_value_0.6.png
# Model 1 name: TeacherModelCNN
# Model: TeacherModel	 Epoch: 1	 Loss: 0.4940	 Accuracy: 0.9287
# Model: TeacherModel	 Epoch: 2	 Loss: 0.2540	 Accuracy: 0.9467
# Model: TeacherModel	 Epoch: 3	 Loss: 0.2061	 Accuracy: 0.9563
# Model: TeacherModel	 Epoch: 4	 Loss: 0.1801	 Accuracy: 0.9562
# Model: TeacherModel	 Epoch: 5	 Loss: 0.1597	 Accuracy: 0.9618
# Model: TeacherModel	 Epoch: 6	 Loss: 0.1470	 Accuracy: 0.9613
# Model: TeacherModel	 Epoch: 7	 Loss: 0.1346	 Accuracy: 0.9674
# Model: TeacherModel	 Epoch: 8	 Loss: 0.1280	 Accuracy: 0.9684
# Model: TeacherModel	 Epoch: 9	 Loss: 0.1192	 Accuracy: 0.9692
# Model: TeacherModel	 Epoch: 10	 Loss: 0.1119	 Accuracy: 0.9728
# Model: TeacherModel	 Epoch: 11	 Loss: 0.1097	 Accuracy: 0.9738
# Model: TeacherModel	 Epoch: 12	 Loss: 0.1050	 Accuracy: 0.9739
# Model: TeacherModel	 Epoch: 13	 Loss: 0.0995	 Accuracy: 0.9738
# Model: TeacherModel	 Epoch: 14	 Loss: 0.0952	 Accuracy: 0.9768
# Model: TeacherModel	 Epoch: 15	 Loss: 0.0903	 Accuracy: 0.9764
# Model: TeacherModel	 Epoch: 16	 Loss: 0.0857	 Accuracy: 0.9761
# Model: TeacherModel	 Epoch: 17	 Loss: 0.0860	 Accuracy: 0.9741
# Model: TeacherModel	 Epoch: 18	 Loss: 0.0849	 Accuracy: 0.9767
# Model: TeacherModel	 Epoch: 19	 Loss: 0.0791	 Accuracy: 0.9788
# Model: TeacherModel	 Epoch: 20	 Loss: 0.0806	 Accuracy: 0.9770
# Model: TeacherModel	 Epoch: 21	 Loss: 0.0760	 Accuracy: 0.9776
# Model: TeacherModel	 Epoch: 22	 Loss: 0.0780	 Accuracy: 0.9788
# Model: TeacherModel	 Epoch: 23	 Loss: 0.0730	 Accuracy: 0.9756
# Model: TeacherModel	 Epoch: 24	 Loss: 0.0747	 Accuracy: 0.9798
# Model: TeacherModel	 Epoch: 25	 Loss: 0.0699	 Accuracy: 0.9782
# Model: TeacherModel	 Epoch: 26	 Loss: 0.0681	 Accuracy: 0.9805
# Model: TeacherModel	 Epoch: 27	 Loss: 0.0645	 Accuracy: 0.9795
# Model: TeacherModel	 Epoch: 28	 Loss: 0.0678	 Accuracy: 0.9803
# Model: TeacherModel	 Epoch: 29	 Loss: 0.0615	 Accuracy: 0.9786
# Model: TeacherModel	 Epoch: 30	 Loss: 0.0611	 Accuracy: 0.9799
# Model: TeacherModel	 Epoch: 31	 Loss: 0.0625	 Accuracy: 0.9800
# Model: TeacherModel	 Epoch: 32	 Loss: 0.0606	 Accuracy: 0.9806
# Model: TeacherModel	 Epoch: 33	 Loss: 0.0582	 Accuracy: 0.9800
# Model: TeacherModel	 Epoch: 34	 Loss: 0.0617	 Accuracy: 0.9802
# Model: TeacherModel	 Epoch: 35	 Loss: 0.0569	 Accuracy: 0.9780
# Model: TeacherModel	 Epoch: 36	 Loss: 0.0548	 Accuracy: 0.9784
# Model: TeacherModel	 Epoch: 37	 Loss: 0.0556	 Accuracy: 0.9807
# Model: TeacherModel	 Epoch: 38	 Loss: 0.0546	 Accuracy: 0.9800
# Model: TeacherModel	 Epoch: 39	 Loss: 0.0545	 Accuracy: 0.9799
# Model: TeacherModel	 Epoch: 40	 Loss: 0.0544	 Accuracy: 0.9803
# Model: TeacherModel	 Epoch: 41	 Loss: 0.0551	 Accuracy: 0.9797
# Model: TeacherModel	 Epoch: 42	 Loss: 0.0541	 Accuracy: 0.9814
# Model: TeacherModel	 Epoch: 43	 Loss: 0.0551	 Accuracy: 0.9801
# Model: TeacherModel	 Epoch: 44	 Loss: 0.0520	 Accuracy: 0.9781
# Model: TeacherModel	 Epoch: 45	 Loss: 0.0487	 Accuracy: 0.9801
# Model: TeacherModel	 Epoch: 46	 Loss: 0.0509	 Accuracy: 0.9812
# Model: TeacherModel	 Epoch: 47	 Loss: 0.0492	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 48	 Loss: 0.0486	 Accuracy: 0.9801
# Model: TeacherModel	 Epoch: 49	 Loss: 0.0472	 Accuracy: 0.9807
# Model: TeacherModel	 Epoch: 50	 Loss: 0.0468	 Accuracy: 0.9817
# Model 2 name: StudentModelA
# Model: StudentModel	 Epoch: 1	 Loss: 1.1623	 Accuracy: 0.8056
# Model: StudentModel	 Epoch: 2	 Loss: 0.5769	 Accuracy: 0.8454
# Model: StudentModel	 Epoch: 3	 Loss: 0.4996	 Accuracy: 0.8526
# Model: StudentModel	 Epoch: 4	 Loss: 0.4641	 Accuracy: 0.8662
# Model: StudentModel	 Epoch: 5	 Loss: 0.4413	 Accuracy: 0.8677
# Model: StudentModel	 Epoch: 6	 Loss: 0.4222	 Accuracy: 0.8738
# Model: StudentModel	 Epoch: 7	 Loss: 0.4058	 Accuracy: 0.8778
# Model: StudentModel	 Epoch: 8	 Loss: 0.3907	 Accuracy: 0.8844
# Model: StudentModel	 Epoch: 9	 Loss: 0.3769	 Accuracy: 0.8878
# Model: StudentModel	 Epoch: 10	 Loss: 0.3668	 Accuracy: 0.8902
# Model: StudentModel	 Epoch: 11	 Loss: 0.3581	 Accuracy: 0.8872
# Model: StudentModel	 Epoch: 12	 Loss: 0.3490	 Accuracy: 0.8948
# Model: StudentModel	 Epoch: 13	 Loss: 0.3429	 Accuracy: 0.8979
# Model: StudentModel	 Epoch: 14	 Loss: 0.3369	 Accuracy: 0.8982
# Model: StudentModel	 Epoch: 15	 Loss: 0.3314	 Accuracy: 0.8998
# Model: StudentModel	 Epoch: 16	 Loss: 0.3254	 Accuracy: 0.9020
# Model: StudentModel	 Epoch: 17	 Loss: 0.3214	 Accuracy: 0.9017
# Model: StudentModel	 Epoch: 18	 Loss: 0.3175	 Accuracy: 0.8992
# Model: StudentModel	 Epoch: 19	 Loss: 0.3143	 Accuracy: 0.9052
# Model: StudentModel	 Epoch: 20	 Loss: 0.3108	 Accuracy: 0.9021
# Model: StudentModel	 Epoch: 21	 Loss: 0.3085	 Accuracy: 0.9074
# Model: StudentModel	 Epoch: 22	 Loss: 0.3056	 Accuracy: 0.9044
# Model: StudentModel	 Epoch: 23	 Loss: 0.3033	 Accuracy: 0.9048
# Model: StudentModel	 Epoch: 24	 Loss: 0.3002	 Accuracy: 0.9042
# Model: StudentModel	 Epoch: 25	 Loss: 0.2977	 Accuracy: 0.9079
# Model: StudentModel	 Epoch: 26	 Loss: 0.2960	 Accuracy: 0.9078
# Model: StudentModel	 Epoch: 27	 Loss: 0.2937	 Accuracy: 0.9107
# Model: StudentModel	 Epoch: 28	 Loss: 0.2920	 Accuracy: 0.9091
# Model: StudentModel	 Epoch: 29	 Loss: 0.2896	 Accuracy: 0.9066
# Model: StudentModel	 Epoch: 30	 Loss: 0.2891	 Accuracy: 0.9083
# Model: StudentModel	 Epoch: 31	 Loss: 0.2868	 Accuracy: 0.9059
# Model: StudentModel	 Epoch: 32	 Loss: 0.2842	 Accuracy: 0.9061
# Model: StudentModel	 Epoch: 33	 Loss: 0.2831	 Accuracy: 0.9093
# Model: StudentModel	 Epoch: 34	 Loss: 0.2805	 Accuracy: 0.9107
# Model: StudentModel	 Epoch: 35	 Loss: 0.2787	 Accuracy: 0.9127
# Model: StudentModel	 Epoch: 36	 Loss: 0.2773	 Accuracy: 0.9113
# Model: StudentModel	 Epoch: 37	 Loss: 0.2743	 Accuracy: 0.9111
# Model: StudentModel	 Epoch: 38	 Loss: 0.2734	 Accuracy: 0.9156
# Model: StudentModel	 Epoch: 39	 Loss: 0.2719	 Accuracy: 0.9132
# Model: StudentModel	 Epoch: 40	 Loss: 0.2702	 Accuracy: 0.9138
# Model: StudentModel	 Epoch: 41	 Loss: 0.2689	 Accuracy: 0.9126
# Model: StudentModel	 Epoch: 42	 Loss: 0.2671	 Accuracy: 0.9143
# Model: StudentModel	 Epoch: 43	 Loss: 0.2658	 Accuracy: 0.9127
# Model: StudentModel	 Epoch: 44	 Loss: 0.2637	 Accuracy: 0.9133
# Model: StudentModel	 Epoch: 45	 Loss: 0.2623	 Accuracy: 0.9131
# Model: StudentModel	 Epoch: 46	 Loss: 0.2611	 Accuracy: 0.9145
# Model: StudentModel	 Epoch: 47	 Loss: 0.2599	 Accuracy: 0.9137
# Model: StudentModel	 Epoch: 48	 Loss: 0.2580	 Accuracy: 0.9156
# Model: StudentModel	 Epoch: 49	 Loss: 0.2568	 Accuracy: 0.9152
# Model: StudentModel	 Epoch: 50	 Loss: 0.2555	 Accuracy: 0.9159
# Model 3 name: StudentModelB
# Model: StudentModel	 Epoch: 1	 Loss: 25.0174	 Accuracy: 0.8573
# Model: StudentModel	 Epoch: 2	 Loss: 20.5236	 Accuracy: 0.8870
# Model: StudentModel	 Epoch: 3	 Loss: 20.0260	 Accuracy: 0.8973
# Model: StudentModel	 Epoch: 4	 Loss: 19.8264	 Accuracy: 0.9038
# Model: StudentModel	 Epoch: 5	 Loss: 19.7004	 Accuracy: 0.9085
# Model: StudentModel	 Epoch: 6	 Loss: 19.6042	 Accuracy: 0.9126
# Model: StudentModel	 Epoch: 7	 Loss: 19.5255	 Accuracy: 0.9143
# Model: StudentModel	 Epoch: 8	 Loss: 19.4586	 Accuracy: 0.9151
# Model: StudentModel	 Epoch: 9	 Loss: 19.4053	 Accuracy: 0.9184
# Model: StudentModel	 Epoch: 10	 Loss: 19.3596	 Accuracy: 0.9194
# Model: StudentModel	 Epoch: 11	 Loss: 19.3146	 Accuracy: 0.9216
# Model: StudentModel	 Epoch: 12	 Loss: 19.2737	 Accuracy: 0.9207
# Model: StudentModel	 Epoch: 13	 Loss: 19.2349	 Accuracy: 0.9224
# Model: StudentModel	 Epoch: 14	 Loss: 19.2019	 Accuracy: 0.9235
# Model: StudentModel	 Epoch: 15	 Loss: 19.1655	 Accuracy: 0.9251
# Model: StudentModel	 Epoch: 16	 Loss: 19.1304	 Accuracy: 0.9289
# Model: StudentModel	 Epoch: 17	 Loss: 19.0989	 Accuracy: 0.9263
# Model: StudentModel	 Epoch: 18	 Loss: 19.0696	 Accuracy: 0.9305
# Model: StudentModel	 Epoch: 19	 Loss: 19.0381	 Accuracy: 0.9313
# Model: StudentModel	 Epoch: 20	 Loss: 19.0111	 Accuracy: 0.9333
# Model: StudentModel	 Epoch: 21	 Loss: 18.9864	 Accuracy: 0.9330
# Model: StudentModel	 Epoch: 22	 Loss: 18.9645	 Accuracy: 0.9332
# Model: StudentModel	 Epoch: 23	 Loss: 18.9399	 Accuracy: 0.9344
# Model: StudentModel	 Epoch: 24	 Loss: 18.9222	 Accuracy: 0.9343
# Model: StudentModel	 Epoch: 25	 Loss: 18.9013	 Accuracy: 0.9377
# Model: StudentModel	 Epoch: 26	 Loss: 18.8843	 Accuracy: 0.9347
# Model: StudentModel	 Epoch: 27	 Loss: 18.8653	 Accuracy: 0.9377
# Model: StudentModel	 Epoch: 28	 Loss: 18.8507	 Accuracy: 0.9368
# Model: StudentModel	 Epoch: 29	 Loss: 18.8360	 Accuracy: 0.9334
# Model: StudentModel	 Epoch: 30	 Loss: 18.8215	 Accuracy: 0.9376
# Model: StudentModel	 Epoch: 31	 Loss: 18.8092	 Accuracy: 0.9389
# Model: StudentModel	 Epoch: 32	 Loss: 18.7959	 Accuracy: 0.9403
# Model: StudentModel	 Epoch: 33	 Loss: 18.7847	 Accuracy: 0.9377
# Model: StudentModel	 Epoch: 34	 Loss: 18.7720	 Accuracy: 0.9389
# Model: StudentModel	 Epoch: 35	 Loss: 18.7595	 Accuracy: 0.9397
# Model: StudentModel	 Epoch: 36	 Loss: 18.7488	 Accuracy: 0.9413
# Model: StudentModel	 Epoch: 37	 Loss: 18.7425	 Accuracy: 0.9407
# Model: StudentModel	 Epoch: 38	 Loss: 18.7311	 Accuracy: 0.9400
# Model: StudentModel	 Epoch: 39	 Loss: 18.7226	 Accuracy: 0.9403
# Model: StudentModel	 Epoch: 40	 Loss: 18.7132	 Accuracy: 0.9406
# Model: StudentModel	 Epoch: 41	 Loss: 18.7052	 Accuracy: 0.9409
# Model: StudentModel	 Epoch: 42	 Loss: 18.6964	 Accuracy: 0.9423
# Model: StudentModel	 Epoch: 43	 Loss: 18.6905	 Accuracy: 0.9423
# Model: StudentModel	 Epoch: 44	 Loss: 18.6814	 Accuracy: 0.9410
# Model: StudentModel	 Epoch: 45	 Loss: 18.6752	 Accuracy: 0.9421
# Model: StudentModel	 Epoch: 46	 Loss: 18.6678	 Accuracy: 0.9409
# Model: StudentModel	 Epoch: 47	 Loss: 18.6637	 Accuracy: 0.9408
# Model: StudentModel	 Epoch: 48	 Loss: 18.6543	 Accuracy: 0.9428
# Model: StudentModel	 Epoch: 49	 Loss: 18.6464	 Accuracy: 0.9442
# Model: StudentModel	 Epoch: 50	 Loss: 18.6410	 Accuracy: 0.9436
# Model 4 name: StudentModelC
# 100%|██████████| 1500/1500 [09:08<00:00,  2.73it/s]
# Model: Kmeans 	 Epoch: 1	 Loss: 0.8573	 Accuracy: 0.8455
# 100%|██████████| 1500/1500 [09:20<00:00,  2.68it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 2	 Loss: 0.7251	 Accuracy: 0.8698
# 100%|██████████| 1500/1500 [10:08<00:00,  2.46it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 3	 Loss: 0.6973	 Accuracy: 0.8778
# 100%|██████████| 1500/1500 [10:55<00:00,  2.29it/s]
# Model: Kmeans 	 Epoch: 4	 Loss: 0.6818	 Accuracy: 0.8825
# 100%|██████████| 1500/1500 [14:46<00:00,  1.69it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 5	 Loss: 0.6710	 Accuracy: 0.8878
# 100%|██████████| 1500/1500 [14:53<00:00,  1.68it/s]
# Model: Kmeans 	 Epoch: 6	 Loss: 0.6628	 Accuracy: 0.8938
# 100%|██████████| 1500/1500 [14:55<00:00,  1.67it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 7	 Loss: 0.6558	 Accuracy: 0.8934
# 100%|██████████| 1500/1500 [13:54<00:00,  1.80it/s]
# Model: Kmeans 	 Epoch: 8	 Loss: 0.6504	 Accuracy: 0.8988
# 100%|██████████| 1500/1500 [13:01<00:00,  1.92it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 9	 Loss: 0.6454	 Accuracy: 0.8992
# 100%|██████████| 1500/1500 [12:32<00:00,  1.99it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 10	 Loss: 0.6412	 Accuracy: 0.9033
# 100%|██████████| 1500/1500 [12:37<00:00,  1.98it/s]
# Model: Kmeans 	 Epoch: 11	 Loss: 0.6376	 Accuracy: 0.9022
# 100%|██████████| 1500/1500 [12:04<00:00,  2.07it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 12	 Loss: 0.6342	 Accuracy: 0.9027
# 100%|██████████| 1500/1500 [12:05<00:00,  2.07it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 13	 Loss: 0.6317	 Accuracy: 0.9048
# 100%|██████████| 1500/1500 [11:48<00:00,  2.12it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 14	 Loss: 0.6294	 Accuracy: 0.9060
# 100%|██████████| 1500/1500 [11:53<00:00,  2.10it/s]
# Model: Kmeans 	 Epoch: 15	 Loss: 0.6272	 Accuracy: 0.9052
# 100%|██████████| 1500/1500 [11:39<00:00,  2.14it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 16	 Loss: 0.6252	 Accuracy: 0.9040
# 100%|██████████| 1500/1500 [11:53<00:00,  2.10it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 17	 Loss: 0.6234	 Accuracy: 0.9077
# 100%|██████████| 1500/1500 [14:10<00:00,  1.76it/s]
# Model: Kmeans 	 Epoch: 18	 Loss: 0.6218	 Accuracy: 0.9072
# 100%|██████████| 1500/1500 [13:12<00:00,  1.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 19	 Loss: 0.6204	 Accuracy: 0.9066
# 100%|██████████| 1500/1500 [13:58<00:00,  1.79it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 20	 Loss: 0.6190	 Accuracy: 0.9061
# 100%|██████████| 1500/1500 [16:33<00:00,  1.51it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 21	 Loss: 0.6180	 Accuracy: 0.9061
# 100%|██████████| 1500/1500 [15:11<00:00,  1.65it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 22	 Loss: 0.6168	 Accuracy: 0.9067
# 100%|██████████| 1500/1500 [17:04<00:00,  1.46it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 23	 Loss: 0.6156	 Accuracy: 0.9058
# 100%|██████████| 1500/1500 [15:03<00:00,  1.66it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 24	 Loss: 0.6144	 Accuracy: 0.9086
# 100%|██████████| 1500/1500 [14:48<00:00,  1.69it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 25	 Loss: 0.6136	 Accuracy: 0.9107
# 100%|██████████| 1500/1500 [15:14<00:00,  1.64it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 26	 Loss: 0.6124	 Accuracy: 0.9091
# 100%|██████████| 1500/1500 [13:23<00:00,  1.87it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 27	 Loss: 0.6115	 Accuracy: 0.9117
# 100%|██████████| 1500/1500 [13:03<00:00,  1.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 28	 Loss: 0.6106	 Accuracy: 0.9091
# 100%|██████████| 1500/1500 [13:55<00:00,  1.80it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 29	 Loss: 0.6096	 Accuracy: 0.9117
# 100%|██████████| 1500/1500 [14:29<00:00,  1.73it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 30	 Loss: 0.6090	 Accuracy: 0.9097
# 100%|██████████| 1500/1500 [14:44<00:00,  1.70it/s]
# Model: Kmeans 	 Epoch: 31	 Loss: 0.6081	 Accuracy: 0.9106
# 100%|██████████| 1500/1500 [14:12<00:00,  1.76it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 32	 Loss: 0.6072	 Accuracy: 0.9097
# 100%|██████████| 1500/1500 [13:55<00:00,  1.80it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 33	 Loss: 0.6064	 Accuracy: 0.9117
# 100%|██████████| 1500/1500 [13:58<00:00,  1.79it/s]
# Model: Kmeans 	 Epoch: 34	 Loss: 0.6056	 Accuracy: 0.9154
# 100%|██████████| 1500/1500 [13:55<00:00,  1.79it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 35	 Loss: 0.6049	 Accuracy: 0.9154
# 100%|██████████| 1500/1500 [13:43<00:00,  1.82it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 36	 Loss: 0.6040	 Accuracy: 0.9155
# 100%|██████████| 1500/1500 [10:29<00:00,  2.38it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 37	 Loss: 0.6031	 Accuracy: 0.9153
# 100%|██████████| 1500/1500 [13:05<00:00,  1.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 38	 Loss: 0.6023	 Accuracy: 0.9158
# 100%|██████████| 1500/1500 [13:18<00:00,  1.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 39	 Loss: 0.6017	 Accuracy: 0.9165
# 100%|██████████| 1500/1500 [12:32<00:00,  1.99it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 40	 Loss: 0.6006	 Accuracy: 0.9174
# 100%|██████████| 1500/1500 [12:35<00:00,  1.99it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 41	 Loss: 0.5997	 Accuracy: 0.9153
# 100%|██████████| 1500/1500 [12:57<00:00,  1.93it/s]
# Model: Kmeans 	 Epoch: 42	 Loss: 0.5990	 Accuracy: 0.9170
# 100%|██████████| 1500/1500 [13:05<00:00,  1.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 43	 Loss: 0.5983	 Accuracy: 0.9164
# 100%|██████████| 1500/1500 [13:08<00:00,  1.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 44	 Loss: 0.5976	 Accuracy: 0.9153
# 100%|██████████| 1500/1500 [13:15<00:00,  1.89it/s]
# Model: Kmeans 	 Epoch: 45	 Loss: 0.5969	 Accuracy: 0.9161
# 100%|██████████| 1500/1500 [11:18<00:00,  2.21it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 46	 Loss: 0.5964	 Accuracy: 0.9173
# 100%|██████████| 1500/1500 [08:44<00:00,  2.86it/s]
# Model: Kmeans 	 Epoch: 47	 Loss: 0.5959	 Accuracy: 0.9160
# 100%|██████████| 1500/1500 [08:42<00:00,  2.87it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 48	 Loss: 0.5952	 Accuracy: 0.9195
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 49	 Loss: 0.5945	 Accuracy: 0.9161
# 100%|██████████| 1500/1500 [08:42<00:00,  2.87it/s]
# Model: Kmeans 	 Epoch: 50	 Loss: 0.5940	 Accuracy: 0.9189
# Save plot as ./alpha_value/alpha_value_0.5.png
# Model 1 name: TeacherModelCNN
# Model: TeacherModel	 Epoch: 1	 Loss: 0.4936	 Accuracy: 0.9250
# Model: TeacherModel	 Epoch: 2	 Loss: 0.2589	 Accuracy: 0.9503
# Model: TeacherModel	 Epoch: 3	 Loss: 0.2047	 Accuracy: 0.9544
# Model: TeacherModel	 Epoch: 4	 Loss: 0.1779	 Accuracy: 0.9609
# Model: TeacherModel	 Epoch: 5	 Loss: 0.1608	 Accuracy: 0.9647
# Model: TeacherModel	 Epoch: 6	 Loss: 0.1449	 Accuracy: 0.9622
# Model: TeacherModel	 Epoch: 7	 Loss: 0.1371	 Accuracy: 0.9685
# Model: TeacherModel	 Epoch: 8	 Loss: 0.1272	 Accuracy: 0.9688
# Model: TeacherModel	 Epoch: 9	 Loss: 0.1223	 Accuracy: 0.9716
# Model: TeacherModel	 Epoch: 10	 Loss: 0.1153	 Accuracy: 0.9730
# Model: TeacherModel	 Epoch: 11	 Loss: 0.1064	 Accuracy: 0.9702
# Model: TeacherModel	 Epoch: 12	 Loss: 0.1023	 Accuracy: 0.9728
# Model: TeacherModel	 Epoch: 13	 Loss: 0.1005	 Accuracy: 0.9742
# Model: TeacherModel	 Epoch: 14	 Loss: 0.0969	 Accuracy: 0.9725
# Model: TeacherModel	 Epoch: 15	 Loss: 0.0924	 Accuracy: 0.9766
# Model: TeacherModel	 Epoch: 16	 Loss: 0.0890	 Accuracy: 0.9742
# Model: TeacherModel	 Epoch: 17	 Loss: 0.0877	 Accuracy: 0.9752
# Model: TeacherModel	 Epoch: 18	 Loss: 0.0836	 Accuracy: 0.9744
# Model: TeacherModel	 Epoch: 19	 Loss: 0.0832	 Accuracy: 0.9756
# Model: TeacherModel	 Epoch: 20	 Loss: 0.0827	 Accuracy: 0.9772
# Model: TeacherModel	 Epoch: 21	 Loss: 0.0779	 Accuracy: 0.9778
# Model: TeacherModel	 Epoch: 22	 Loss: 0.0766	 Accuracy: 0.9778
# Model: TeacherModel	 Epoch: 23	 Loss: 0.0753	 Accuracy: 0.9782
# Model: TeacherModel	 Epoch: 24	 Loss: 0.0710	 Accuracy: 0.9790
# Model: TeacherModel	 Epoch: 25	 Loss: 0.0677	 Accuracy: 0.9805
# Model: TeacherModel	 Epoch: 26	 Loss: 0.0696	 Accuracy: 0.9782
# Model: TeacherModel	 Epoch: 27	 Loss: 0.0678	 Accuracy: 0.9788
# Model: TeacherModel	 Epoch: 28	 Loss: 0.0672	 Accuracy: 0.9806
# Model: TeacherModel	 Epoch: 29	 Loss: 0.0642	 Accuracy: 0.9810
# Model: TeacherModel	 Epoch: 30	 Loss: 0.0631	 Accuracy: 0.9772
# Model: TeacherModel	 Epoch: 31	 Loss: 0.0582	 Accuracy: 0.9809
# Model: TeacherModel	 Epoch: 32	 Loss: 0.0625	 Accuracy: 0.9806
# Model: TeacherModel	 Epoch: 33	 Loss: 0.0592	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 34	 Loss: 0.0620	 Accuracy: 0.9803
# Model: TeacherModel	 Epoch: 35	 Loss: 0.0581	 Accuracy: 0.9794
# Model: TeacherModel	 Epoch: 36	 Loss: 0.0574	 Accuracy: 0.9814
# Model: TeacherModel	 Epoch: 37	 Loss: 0.0560	 Accuracy: 0.9805
# Model: TeacherModel	 Epoch: 38	 Loss: 0.0542	 Accuracy: 0.9807
# Model: TeacherModel	 Epoch: 39	 Loss: 0.0545	 Accuracy: 0.9798
# Model: TeacherModel	 Epoch: 40	 Loss: 0.0560	 Accuracy: 0.9811
# Model: TeacherModel	 Epoch: 41	 Loss: 0.0548	 Accuracy: 0.9790
# Model: TeacherModel	 Epoch: 42	 Loss: 0.0514	 Accuracy: 0.9813
# Model: TeacherModel	 Epoch: 43	 Loss: 0.0542	 Accuracy: 0.9820
# Model: TeacherModel	 Epoch: 44	 Loss: 0.0486	 Accuracy: 0.9801
# Model: TeacherModel	 Epoch: 45	 Loss: 0.0522	 Accuracy: 0.9804
# Model: TeacherModel	 Epoch: 46	 Loss: 0.0520	 Accuracy: 0.9824
# Model: TeacherModel	 Epoch: 47	 Loss: 0.0481	 Accuracy: 0.9818
# Model: TeacherModel	 Epoch: 48	 Loss: 0.0478	 Accuracy: 0.9818
# Model: TeacherModel	 Epoch: 49	 Loss: 0.0480	 Accuracy: 0.9818
# Model: TeacherModel	 Epoch: 50	 Loss: 0.0484	 Accuracy: 0.9807
# Model 2 name: StudentModelA
# Model: StudentModel	 Epoch: 1	 Loss: 0.9753	 Accuracy: 0.8539
# Model: StudentModel	 Epoch: 2	 Loss: 0.4475	 Accuracy: 0.8828
# Model: StudentModel	 Epoch: 3	 Loss: 0.3755	 Accuracy: 0.8948
# Model: StudentModel	 Epoch: 4	 Loss: 0.3447	 Accuracy: 0.8969
# Model: StudentModel	 Epoch: 5	 Loss: 0.3253	 Accuracy: 0.8992
# Model: StudentModel	 Epoch: 6	 Loss: 0.3115	 Accuracy: 0.9057
# Model: StudentModel	 Epoch: 7	 Loss: 0.3001	 Accuracy: 0.9091
# Model: StudentModel	 Epoch: 8	 Loss: 0.2905	 Accuracy: 0.9095
# Model: StudentModel	 Epoch: 9	 Loss: 0.2825	 Accuracy: 0.9133
# Model: StudentModel	 Epoch: 10	 Loss: 0.2759	 Accuracy: 0.9108
# Model: StudentModel	 Epoch: 11	 Loss: 0.2694	 Accuracy: 0.9156
# Model: StudentModel	 Epoch: 12	 Loss: 0.2646	 Accuracy: 0.9133
# Model: StudentModel	 Epoch: 13	 Loss: 0.2592	 Accuracy: 0.9173
# Model: StudentModel	 Epoch: 14	 Loss: 0.2554	 Accuracy: 0.9209
# Model: StudentModel	 Epoch: 15	 Loss: 0.2508	 Accuracy: 0.9202
# Model: StudentModel	 Epoch: 16	 Loss: 0.2470	 Accuracy: 0.9213
# Model: StudentModel	 Epoch: 17	 Loss: 0.2438	 Accuracy: 0.9191
# Model: StudentModel	 Epoch: 18	 Loss: 0.2407	 Accuracy: 0.9204
# Model: StudentModel	 Epoch: 19	 Loss: 0.2378	 Accuracy: 0.9208
# Model: StudentModel	 Epoch: 20	 Loss: 0.2353	 Accuracy: 0.9213
# Model: StudentModel	 Epoch: 21	 Loss: 0.2330	 Accuracy: 0.9228
# Model: StudentModel	 Epoch: 22	 Loss: 0.2298	 Accuracy: 0.9255
# Model: StudentModel	 Epoch: 23	 Loss: 0.2282	 Accuracy: 0.9252
# Model: StudentModel	 Epoch: 24	 Loss: 0.2253	 Accuracy: 0.9247
# Model: StudentModel	 Epoch: 25	 Loss: 0.2241	 Accuracy: 0.9256
# Model: StudentModel	 Epoch: 26	 Loss: 0.2222	 Accuracy: 0.9236
# Model: StudentModel	 Epoch: 27	 Loss: 0.2201	 Accuracy: 0.9263
# Model: StudentModel	 Epoch: 28	 Loss: 0.2189	 Accuracy: 0.9262
# Model: StudentModel	 Epoch: 29	 Loss: 0.2164	 Accuracy: 0.9270
# Model: StudentModel	 Epoch: 30	 Loss: 0.2149	 Accuracy: 0.9282
# Model: StudentModel	 Epoch: 31	 Loss: 0.2133	 Accuracy: 0.9297
# Model: StudentModel	 Epoch: 32	 Loss: 0.2121	 Accuracy: 0.9297
# Model: StudentModel	 Epoch: 33	 Loss: 0.2108	 Accuracy: 0.9302
# Model: StudentModel	 Epoch: 34	 Loss: 0.2088	 Accuracy: 0.9273
# Model: StudentModel	 Epoch: 35	 Loss: 0.2079	 Accuracy: 0.9276
# Model: StudentModel	 Epoch: 36	 Loss: 0.2063	 Accuracy: 0.9313
# Model: StudentModel	 Epoch: 37	 Loss: 0.2054	 Accuracy: 0.9303
# Model: StudentModel	 Epoch: 38	 Loss: 0.2041	 Accuracy: 0.9289
# Model: StudentModel	 Epoch: 39	 Loss: 0.2028	 Accuracy: 0.9309
# Model: StudentModel	 Epoch: 40	 Loss: 0.2016	 Accuracy: 0.9314
# Model: StudentModel	 Epoch: 41	 Loss: 0.2008	 Accuracy: 0.9304
# Model: StudentModel	 Epoch: 42	 Loss: 0.2001	 Accuracy: 0.9310
# Model: StudentModel	 Epoch: 43	 Loss: 0.1983	 Accuracy: 0.9313
# Model: StudentModel	 Epoch: 44	 Loss: 0.1975	 Accuracy: 0.9303
# Model: StudentModel	 Epoch: 45	 Loss: 0.1972	 Accuracy: 0.9324
# Model: StudentModel	 Epoch: 46	 Loss: 0.1956	 Accuracy: 0.9307
# Model: StudentModel	 Epoch: 47	 Loss: 0.1950	 Accuracy: 0.9314
# Model: StudentModel	 Epoch: 48	 Loss: 0.1942	 Accuracy: 0.9337
# Model: StudentModel	 Epoch: 49	 Loss: 0.1934	 Accuracy: 0.9322
# Model: StudentModel	 Epoch: 50	 Loss: 0.1923	 Accuracy: 0.9345
# Model 3 name: StudentModelB
# Model: StudentModel	 Epoch: 1	 Loss: 24.9047	 Accuracy: 0.8604
# Model: StudentModel	 Epoch: 2	 Loss: 20.4944	 Accuracy: 0.8887
# Model: StudentModel	 Epoch: 3	 Loss: 20.0000	 Accuracy: 0.8973
# Model: StudentModel	 Epoch: 4	 Loss: 19.7967	 Accuracy: 0.9031
# Model: StudentModel	 Epoch: 5	 Loss: 19.6621	 Accuracy: 0.9071
# Model: StudentModel	 Epoch: 6	 Loss: 19.5569	 Accuracy: 0.9123
# Model: StudentModel	 Epoch: 7	 Loss: 19.4695	 Accuracy: 0.9109
# Model: StudentModel	 Epoch: 8	 Loss: 19.3957	 Accuracy: 0.9133
# Model: StudentModel	 Epoch: 9	 Loss: 19.3332	 Accuracy: 0.9187
# Model: StudentModel	 Epoch: 10	 Loss: 19.2783	 Accuracy: 0.9197
# Model: StudentModel	 Epoch: 11	 Loss: 19.2291	 Accuracy: 0.9214
# Model: StudentModel	 Epoch: 12	 Loss: 19.1862	 Accuracy: 0.9204
# Model: StudentModel	 Epoch: 13	 Loss: 19.1417	 Accuracy: 0.9237
# Model: StudentModel	 Epoch: 14	 Loss: 19.1042	 Accuracy: 0.9246
# Model: StudentModel	 Epoch: 15	 Loss: 19.0644	 Accuracy: 0.9254
# Model: StudentModel	 Epoch: 16	 Loss: 19.0344	 Accuracy: 0.9260
# Model: StudentModel	 Epoch: 17	 Loss: 19.0049	 Accuracy: 0.9291
# Model: StudentModel	 Epoch: 18	 Loss: 18.9755	 Accuracy: 0.9291
# Model: StudentModel	 Epoch: 19	 Loss: 18.9457	 Accuracy: 0.9287
# Model: StudentModel	 Epoch: 20	 Loss: 18.9268	 Accuracy: 0.9328
# Model: StudentModel	 Epoch: 21	 Loss: 18.8987	 Accuracy: 0.9342
# Model: StudentModel	 Epoch: 22	 Loss: 18.8792	 Accuracy: 0.9326
# Model: StudentModel	 Epoch: 23	 Loss: 18.8523	 Accuracy: 0.9328
# Model: StudentModel	 Epoch: 24	 Loss: 18.8350	 Accuracy: 0.9363
# Model: StudentModel	 Epoch: 25	 Loss: 18.8152	 Accuracy: 0.9350
# Model: StudentModel	 Epoch: 26	 Loss: 18.7961	 Accuracy: 0.9353
# Model: StudentModel	 Epoch: 27	 Loss: 18.7772	 Accuracy: 0.9364
# Model: StudentModel	 Epoch: 28	 Loss: 18.7605	 Accuracy: 0.9344
# Model: StudentModel	 Epoch: 29	 Loss: 18.7435	 Accuracy: 0.9381
# Model: StudentModel	 Epoch: 30	 Loss: 18.7288	 Accuracy: 0.9402
# Model: StudentModel	 Epoch: 31	 Loss: 18.7158	 Accuracy: 0.9390
# Model: StudentModel	 Epoch: 32	 Loss: 18.7005	 Accuracy: 0.9403
# Model: StudentModel	 Epoch: 33	 Loss: 18.6865	 Accuracy: 0.9400
# Model: StudentModel	 Epoch: 34	 Loss: 18.6727	 Accuracy: 0.9383
# Model: StudentModel	 Epoch: 35	 Loss: 18.6594	 Accuracy: 0.9431
# Model: StudentModel	 Epoch: 36	 Loss: 18.6488	 Accuracy: 0.9399
# Model: StudentModel	 Epoch: 37	 Loss: 18.6387	 Accuracy: 0.9396
# Model: StudentModel	 Epoch: 38	 Loss: 18.6303	 Accuracy: 0.9413
# Model: StudentModel	 Epoch: 39	 Loss: 18.6185	 Accuracy: 0.9407
# Model: StudentModel	 Epoch: 40	 Loss: 18.6107	 Accuracy: 0.9416
# Model: StudentModel	 Epoch: 41	 Loss: 18.6008	 Accuracy: 0.9405
# Model: StudentModel	 Epoch: 42	 Loss: 18.5936	 Accuracy: 0.9434
# Model: StudentModel	 Epoch: 43	 Loss: 18.5844	 Accuracy: 0.9424
# Model: StudentModel	 Epoch: 44	 Loss: 18.5743	 Accuracy: 0.9404
# Model: StudentModel	 Epoch: 45	 Loss: 18.5695	 Accuracy: 0.9433
# Model: StudentModel	 Epoch: 46	 Loss: 18.5628	 Accuracy: 0.9413
# Model: StudentModel	 Epoch: 47	 Loss: 18.5545	 Accuracy: 0.9443
# Model: StudentModel	 Epoch: 48	 Loss: 18.5462	 Accuracy: 0.9443
# Model: StudentModel	 Epoch: 49	 Loss: 18.5402	 Accuracy: 0.9441
# Model: StudentModel	 Epoch: 50	 Loss: 18.5363	 Accuracy: 0.9427
# Model 4 name: StudentModelC
# 100%|██████████| 1500/1500 [08:44<00:00,  2.86it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 1	 Loss: 0.7773	 Accuracy: 0.7533
# 100%|██████████| 1500/1500 [08:45<00:00,  2.85it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 2	 Loss: 0.6665	 Accuracy: 0.8284
# 100%|██████████| 1500/1500 [08:43<00:00,  2.87it/s]
# Model: Kmeans 	 Epoch: 3	 Loss: 0.6441	 Accuracy: 0.8511
# 100%|██████████| 1500/1500 [08:46<00:00,  2.85it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 4	 Loss: 0.6320	 Accuracy: 0.8569
# 100%|██████████| 1500/1500 [08:43<00:00,  2.86it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 5	 Loss: 0.6231	 Accuracy: 0.8600
# 100%|██████████| 1500/1500 [08:41<00:00,  2.87it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 6	 Loss: 0.6166	 Accuracy: 0.8706
# 100%|██████████| 1500/1500 [08:44<00:00,  2.86it/s]
# Model: Kmeans 	 Epoch: 7	 Loss: 0.6117	 Accuracy: 0.8764
# 100%|██████████| 1500/1500 [08:44<00:00,  2.86it/s]
# Model: Kmeans 	 Epoch: 8	 Loss: 0.6077	 Accuracy: 0.8775
# 100%|██████████| 1500/1500 [08:44<00:00,  2.86it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 9	 Loss: 0.6041	 Accuracy: 0.8760
# 100%|██████████| 1500/1500 [08:40<00:00,  2.88it/s]
# Model: Kmeans 	 Epoch: 10	 Loss: 0.6013	 Accuracy: 0.8798
# 100%|██████████| 1500/1500 [08:41<00:00,  2.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 11	 Loss: 0.5988	 Accuracy: 0.8824
# 100%|██████████| 1500/1500 [08:40<00:00,  2.88it/s]
# Model: Kmeans 	 Epoch: 12	 Loss: 0.5965	 Accuracy: 0.8817
# 100%|██████████| 1500/1500 [08:40<00:00,  2.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 13	 Loss: 0.5945	 Accuracy: 0.8861
# 100%|██████████| 1500/1500 [08:45<00:00,  2.85it/s]
# Model: Kmeans 	 Epoch: 14	 Loss: 0.5929	 Accuracy: 0.8832
# 100%|██████████| 1500/1500 [08:44<00:00,  2.86it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 15	 Loss: 0.5910	 Accuracy: 0.8892
# 100%|██████████| 1500/1500 [08:42<00:00,  2.87it/s]
# Model: Kmeans 	 Epoch: 16	 Loss: 0.5895	 Accuracy: 0.8915
# 100%|██████████| 1500/1500 [08:42<00:00,  2.87it/s]
# Model: Kmeans 	 Epoch: 17	 Loss: 0.5882	 Accuracy: 0.8898
# 100%|██████████| 1500/1500 [08:46<00:00,  2.85it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 18	 Loss: 0.5870	 Accuracy: 0.8940
# 100%|██████████| 1500/1500 [08:39<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 19	 Loss: 0.5861	 Accuracy: 0.8932
# 100%|██████████| 1500/1500 [08:45<00:00,  2.85it/s]
# Model: Kmeans 	 Epoch: 20	 Loss: 0.5847	 Accuracy: 0.8925
# 100%|██████████| 1500/1500 [08:41<00:00,  2.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 21	 Loss: 0.5839	 Accuracy: 0.8971
# 100%|██████████| 1500/1500 [08:42<00:00,  2.87it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 22	 Loss: 0.5831	 Accuracy: 0.8961
# 100%|██████████| 1500/1500 [08:43<00:00,  2.87it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 23	 Loss: 0.5822	 Accuracy: 0.8963
# 100%|██████████| 1500/1500 [08:41<00:00,  2.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 24	 Loss: 0.5813	 Accuracy: 0.8967
# 100%|██████████| 1500/1500 [08:41<00:00,  2.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 25	 Loss: 0.5804	 Accuracy: 0.8937
# 100%|██████████| 1500/1500 [08:39<00:00,  2.89it/s]
# Model: Kmeans 	 Epoch: 26	 Loss: 0.5796	 Accuracy: 0.8968
# 100%|██████████| 1500/1500 [08:42<00:00,  2.87it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 27	 Loss: 0.5790	 Accuracy: 0.8964
# 100%|██████████| 1500/1500 [08:40<00:00,  2.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 28	 Loss: 0.5784	 Accuracy: 0.8976
# 100%|██████████| 1500/1500 [08:38<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 29	 Loss: 0.5777	 Accuracy: 0.8997
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 30	 Loss: 0.5770	 Accuracy: 0.8962
# 100%|██████████| 1500/1500 [08:40<00:00,  2.88it/s]
# Model: Kmeans 	 Epoch: 31	 Loss: 0.5765	 Accuracy: 0.8971
# 100%|██████████| 1500/1500 [08:40<00:00,  2.88it/s]
# Model: Kmeans 	 Epoch: 32	 Loss: 0.5759	 Accuracy: 0.8970
# 100%|██████████| 1500/1500 [08:40<00:00,  2.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 33	 Loss: 0.5755	 Accuracy: 0.9009
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
# Model: Kmeans 	 Epoch: 34	 Loss: 0.5749	 Accuracy: 0.9029
# 100%|██████████| 1500/1500 [08:40<00:00,  2.88it/s]
# Model: Kmeans 	 Epoch: 35	 Loss: 0.5743	 Accuracy: 0.8987
# 100%|██████████| 1500/1500 [08:38<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 36	 Loss: 0.5738	 Accuracy: 0.8927
# 100%|██████████| 1500/1500 [08:36<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 37	 Loss: 0.5735	 Accuracy: 0.9022
# 100%|██████████| 1500/1500 [08:39<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 38	 Loss: 0.5731	 Accuracy: 0.8978
# 100%|██████████| 1500/1500 [08:34<00:00,  2.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 39	 Loss: 0.5725	 Accuracy: 0.9008
# 100%|██████████| 1500/1500 [08:36<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 40	 Loss: 0.5721	 Accuracy: 0.9008
# 100%|██████████| 1500/1500 [08:36<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 41	 Loss: 0.5717	 Accuracy: 0.9040
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 42	 Loss: 0.5714	 Accuracy: 0.9026
# 100%|██████████| 1500/1500 [08:35<00:00,  2.91it/s]
# Model: Kmeans 	 Epoch: 43	 Loss: 0.5709	 Accuracy: 0.8969
# 100%|██████████| 1500/1500 [08:35<00:00,  2.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 44	 Loss: 0.5705	 Accuracy: 0.8987
# 100%|██████████| 1500/1500 [08:40<00:00,  2.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 45	 Loss: 0.5702	 Accuracy: 0.9022
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 46	 Loss: 0.5697	 Accuracy: 0.9028
# 100%|██████████| 1500/1500 [08:34<00:00,  2.92it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 47	 Loss: 0.5695	 Accuracy: 0.8999
# 100%|██████████| 1500/1500 [08:30<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 48	 Loss: 0.5690	 Accuracy: 0.8978
# 100%|██████████| 1500/1500 [08:39<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 49	 Loss: 0.5689	 Accuracy: 0.9010
# 100%|██████████| 1500/1500 [08:36<00:00,  2.90it/s]
# Model: Kmeans 	 Epoch: 50	 Loss: 0.5683	 Accuracy: 0.9007
# Save plot as ./alpha_value/alpha_value_0.4.png
# Model 1 name: TeacherModelCNN
# Model: TeacherModel	 Epoch: 1	 Loss: 0.4984	 Accuracy: 0.9234
# Model: TeacherModel	 Epoch: 2	 Loss: 0.2572	 Accuracy: 0.9467
# Model: TeacherModel	 Epoch: 3	 Loss: 0.2070	 Accuracy: 0.9467
# Model: TeacherModel	 Epoch: 4	 Loss: 0.1764	 Accuracy: 0.9552
# Model: TeacherModel	 Epoch: 5	 Loss: 0.1585	 Accuracy: 0.9616
# Model: TeacherModel	 Epoch: 6	 Loss: 0.1466	 Accuracy: 0.9663
# Model: TeacherModel	 Epoch: 7	 Loss: 0.1372	 Accuracy: 0.9657
# Model: TeacherModel	 Epoch: 8	 Loss: 0.1282	 Accuracy: 0.9702
# Model: TeacherModel	 Epoch: 9	 Loss: 0.1192	 Accuracy: 0.9711
# Model: TeacherModel	 Epoch: 10	 Loss: 0.1154	 Accuracy: 0.9696
# Model: TeacherModel	 Epoch: 11	 Loss: 0.1069	 Accuracy: 0.9693
# Model: TeacherModel	 Epoch: 12	 Loss: 0.1041	 Accuracy: 0.9761
# Model: TeacherModel	 Epoch: 13	 Loss: 0.0995	 Accuracy: 0.9743
# Model: TeacherModel	 Epoch: 14	 Loss: 0.0948	 Accuracy: 0.9748
# Model: TeacherModel	 Epoch: 15	 Loss: 0.0933	 Accuracy: 0.9740
# Model: TeacherModel	 Epoch: 16	 Loss: 0.0906	 Accuracy: 0.9764
# Model: TeacherModel	 Epoch: 17	 Loss: 0.0848	 Accuracy: 0.9724
# Model: TeacherModel	 Epoch: 18	 Loss: 0.0831	 Accuracy: 0.9762
# Model: TeacherModel	 Epoch: 19	 Loss: 0.0837	 Accuracy: 0.9772
# Model: TeacherModel	 Epoch: 20	 Loss: 0.0776	 Accuracy: 0.9772
# Model: TeacherModel	 Epoch: 21	 Loss: 0.0743	 Accuracy: 0.9784
# Model: TeacherModel	 Epoch: 22	 Loss: 0.0781	 Accuracy: 0.9792
# Model: TeacherModel	 Epoch: 23	 Loss: 0.0730	 Accuracy: 0.9782
# Model: TeacherModel	 Epoch: 24	 Loss: 0.0728	 Accuracy: 0.9788
# Model: TeacherModel	 Epoch: 25	 Loss: 0.0721	 Accuracy: 0.9800
# Model: TeacherModel	 Epoch: 26	 Loss: 0.0665	 Accuracy: 0.9787
# Model: TeacherModel	 Epoch: 27	 Loss: 0.0684	 Accuracy: 0.9802
# Model: TeacherModel	 Epoch: 28	 Loss: 0.0685	 Accuracy: 0.9794
# Model: TeacherModel	 Epoch: 29	 Loss: 0.0634	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 30	 Loss: 0.0604	 Accuracy: 0.9800
# Model: TeacherModel	 Epoch: 31	 Loss: 0.0633	 Accuracy: 0.9798
# Model: TeacherModel	 Epoch: 32	 Loss: 0.0596	 Accuracy: 0.9813
# Model: TeacherModel	 Epoch: 33	 Loss: 0.0596	 Accuracy: 0.9804
# Model: TeacherModel	 Epoch: 34	 Loss: 0.0589	 Accuracy: 0.9804
# Model: TeacherModel	 Epoch: 35	 Loss: 0.0572	 Accuracy: 0.9809
# Model: TeacherModel	 Epoch: 36	 Loss: 0.0573	 Accuracy: 0.9815
# Model: TeacherModel	 Epoch: 37	 Loss: 0.0552	 Accuracy: 0.9816
# Model: TeacherModel	 Epoch: 38	 Loss: 0.0567	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 39	 Loss: 0.0544	 Accuracy: 0.9807
# Model: TeacherModel	 Epoch: 40	 Loss: 0.0550	 Accuracy: 0.9811
# Model: TeacherModel	 Epoch: 41	 Loss: 0.0537	 Accuracy: 0.9805
# Model: TeacherModel	 Epoch: 42	 Loss: 0.0531	 Accuracy: 0.9810
# Model: TeacherModel	 Epoch: 43	 Loss: 0.0538	 Accuracy: 0.9807
# Model: TeacherModel	 Epoch: 44	 Loss: 0.0513	 Accuracy: 0.9824
# Model: TeacherModel	 Epoch: 45	 Loss: 0.0516	 Accuracy: 0.9812
# Model: TeacherModel	 Epoch: 46	 Loss: 0.0471	 Accuracy: 0.9816
# Model: TeacherModel	 Epoch: 47	 Loss: 0.0502	 Accuracy: 0.9821
# Model: TeacherModel	 Epoch: 48	 Loss: 0.0481	 Accuracy: 0.9824
# Model: TeacherModel	 Epoch: 49	 Loss: 0.0493	 Accuracy: 0.9815
# Model: TeacherModel	 Epoch: 50	 Loss: 0.0493	 Accuracy: 0.9802
# Model 2 name: StudentModelA
# Model: StudentModel	 Epoch: 1	 Loss: 0.8673	 Accuracy: 0.8604
# Model: StudentModel	 Epoch: 2	 Loss: 0.4189	 Accuracy: 0.8906
# Model: StudentModel	 Epoch: 3	 Loss: 0.3530	 Accuracy: 0.9005
# Model: StudentModel	 Epoch: 4	 Loss: 0.3238	 Accuracy: 0.9009
# Model: StudentModel	 Epoch: 5	 Loss: 0.3060	 Accuracy: 0.9057
# Model: StudentModel	 Epoch: 6	 Loss: 0.2919	 Accuracy: 0.9100
# Model: StudentModel	 Epoch: 7	 Loss: 0.2807	 Accuracy: 0.9085
# Model: StudentModel	 Epoch: 8	 Loss: 0.2712	 Accuracy: 0.9150
# Model: StudentModel	 Epoch: 9	 Loss: 0.2635	 Accuracy: 0.9183
# Model: StudentModel	 Epoch: 10	 Loss: 0.2566	 Accuracy: 0.9211
# Model: StudentModel	 Epoch: 11	 Loss: 0.2515	 Accuracy: 0.9227
# Model: StudentModel	 Epoch: 12	 Loss: 0.2454	 Accuracy: 0.9244
# Model: StudentModel	 Epoch: 13	 Loss: 0.2400	 Accuracy: 0.9247
# Model: StudentModel	 Epoch: 14	 Loss: 0.2352	 Accuracy: 0.9209
# Model: StudentModel	 Epoch: 15	 Loss: 0.2311	 Accuracy: 0.9235
# Model: StudentModel	 Epoch: 16	 Loss: 0.2263	 Accuracy: 0.9279
# Model: StudentModel	 Epoch: 17	 Loss: 0.2236	 Accuracy: 0.9277
# Model: StudentModel	 Epoch: 18	 Loss: 0.2183	 Accuracy: 0.9284
# Model: StudentModel	 Epoch: 19	 Loss: 0.2153	 Accuracy: 0.9275
# Model: StudentModel	 Epoch: 20	 Loss: 0.2122	 Accuracy: 0.9313
# Model: StudentModel	 Epoch: 21	 Loss: 0.2095	 Accuracy: 0.9293
# Model: StudentModel	 Epoch: 22	 Loss: 0.2067	 Accuracy: 0.9324
# Model: StudentModel	 Epoch: 23	 Loss: 0.2035	 Accuracy: 0.9327
# Model: StudentModel	 Epoch: 24	 Loss: 0.2017	 Accuracy: 0.9339
# Model: StudentModel	 Epoch: 25	 Loss: 0.1987	 Accuracy: 0.9307
# Model: StudentModel	 Epoch: 26	 Loss: 0.1962	 Accuracy: 0.9319
# Model: StudentModel	 Epoch: 27	 Loss: 0.1948	 Accuracy: 0.9322
# Model: StudentModel	 Epoch: 28	 Loss: 0.1919	 Accuracy: 0.9348
# Model: StudentModel	 Epoch: 29	 Loss: 0.1907	 Accuracy: 0.9344
# Model: StudentModel	 Epoch: 30	 Loss: 0.1881	 Accuracy: 0.9353
# Model: StudentModel	 Epoch: 31	 Loss: 0.1863	 Accuracy: 0.9339
# Model: StudentModel	 Epoch: 32	 Loss: 0.1855	 Accuracy: 0.9370
# Model: StudentModel	 Epoch: 33	 Loss: 0.1835	 Accuracy: 0.9368
# Model: StudentModel	 Epoch: 34	 Loss: 0.1809	 Accuracy: 0.9343
# Model: StudentModel	 Epoch: 35	 Loss: 0.1806	 Accuracy: 0.9337
# Model: StudentModel	 Epoch: 36	 Loss: 0.1796	 Accuracy: 0.9368
# Model: StudentModel	 Epoch: 37	 Loss: 0.1773	 Accuracy: 0.9373
# Model: StudentModel	 Epoch: 38	 Loss: 0.1763	 Accuracy: 0.9364
# Model: StudentModel	 Epoch: 39	 Loss: 0.1748	 Accuracy: 0.9348
# Model: StudentModel	 Epoch: 40	 Loss: 0.1734	 Accuracy: 0.9357
# Model: StudentModel	 Epoch: 41	 Loss: 0.1728	 Accuracy: 0.9377
# Model: StudentModel	 Epoch: 42	 Loss: 0.1711	 Accuracy: 0.9386
# Model: StudentModel	 Epoch: 43	 Loss: 0.1700	 Accuracy: 0.9373
# Model: StudentModel	 Epoch: 44	 Loss: 0.1690	 Accuracy: 0.9372
# Model: StudentModel	 Epoch: 45	 Loss: 0.1677	 Accuracy: 0.9388
# Model: StudentModel	 Epoch: 46	 Loss: 0.1665	 Accuracy: 0.9359
# Model: StudentModel	 Epoch: 47	 Loss: 0.1663	 Accuracy: 0.9396
# Model: StudentModel	 Epoch: 48	 Loss: 0.1647	 Accuracy: 0.9404
# Model: StudentModel	 Epoch: 49	 Loss: 0.1637	 Accuracy: 0.9389
# Model: StudentModel	 Epoch: 50	 Loss: 0.1630	 Accuracy: 0.9391
# Model 3 name: StudentModelB
# Model: StudentModel	 Epoch: 1	 Loss: 26.2996	 Accuracy: 0.7928
# Model: StudentModel	 Epoch: 2	 Loss: 21.6341	 Accuracy: 0.8552
# Model: StudentModel	 Epoch: 3	 Loss: 20.6768	 Accuracy: 0.8768
# Model: StudentModel	 Epoch: 4	 Loss: 20.2536	 Accuracy: 0.8878
# Model: StudentModel	 Epoch: 5	 Loss: 20.0477	 Accuracy: 0.8888
# Model: StudentModel	 Epoch: 6	 Loss: 19.9206	 Accuracy: 0.8920
# Model: StudentModel	 Epoch: 7	 Loss: 19.8272	 Accuracy: 0.8998
# Model: StudentModel	 Epoch: 8	 Loss: 19.7490	 Accuracy: 0.9016
# Model: StudentModel	 Epoch: 9	 Loss: 19.6818	 Accuracy: 0.9039
# Model: StudentModel	 Epoch: 10	 Loss: 19.6266	 Accuracy: 0.9055
# Model: StudentModel	 Epoch: 11	 Loss: 19.5805	 Accuracy: 0.9067
# Model: StudentModel	 Epoch: 12	 Loss: 19.5386	 Accuracy: 0.9070
# Model: StudentModel	 Epoch: 13	 Loss: 19.5030	 Accuracy: 0.9080
# Model: StudentModel	 Epoch: 14	 Loss: 19.4723	 Accuracy: 0.9115
# Model: StudentModel	 Epoch: 15	 Loss: 19.4411	 Accuracy: 0.9101
# Model: StudentModel	 Epoch: 16	 Loss: 19.4102	 Accuracy: 0.9105
# Model: StudentModel	 Epoch: 17	 Loss: 19.3834	 Accuracy: 0.9137
# Model: StudentModel	 Epoch: 18	 Loss: 19.3585	 Accuracy: 0.9131
# Model: StudentModel	 Epoch: 19	 Loss: 19.3312	 Accuracy: 0.9132
# Model: StudentModel	 Epoch: 20	 Loss: 19.3081	 Accuracy: 0.9140
# Model: StudentModel	 Epoch: 21	 Loss: 19.2836	 Accuracy: 0.9157
# Model: StudentModel	 Epoch: 22	 Loss: 19.2609	 Accuracy: 0.9167
# Model: StudentModel	 Epoch: 23	 Loss: 19.2394	 Accuracy: 0.9174
# Model: StudentModel	 Epoch: 24	 Loss: 19.2167	 Accuracy: 0.9170
# Model: StudentModel	 Epoch: 25	 Loss: 19.1965	 Accuracy: 0.9194
# Model: StudentModel	 Epoch: 26	 Loss: 19.1752	 Accuracy: 0.9206
# Model: StudentModel	 Epoch: 27	 Loss: 19.1555	 Accuracy: 0.9213
# Model: StudentModel	 Epoch: 28	 Loss: 19.1383	 Accuracy: 0.9215
# Model: StudentModel	 Epoch: 29	 Loss: 19.1205	 Accuracy: 0.9223
# Model: StudentModel	 Epoch: 30	 Loss: 19.1046	 Accuracy: 0.9229
# Model: StudentModel	 Epoch: 31	 Loss: 19.0862	 Accuracy: 0.9266
# Model: StudentModel	 Epoch: 32	 Loss: 19.0740	 Accuracy: 0.9236
# Model: StudentModel	 Epoch: 33	 Loss: 19.0597	 Accuracy: 0.9257
# Model: StudentModel	 Epoch: 34	 Loss: 19.0482	 Accuracy: 0.9281
# Model: StudentModel	 Epoch: 35	 Loss: 19.0362	 Accuracy: 0.9262
# Model: StudentModel	 Epoch: 36	 Loss: 19.0242	 Accuracy: 0.9258
# Model: StudentModel	 Epoch: 37	 Loss: 19.0120	 Accuracy: 0.9253
# Model: StudentModel	 Epoch: 38	 Loss: 19.0037	 Accuracy: 0.9274
# Model: StudentModel	 Epoch: 39	 Loss: 18.9943	 Accuracy: 0.9281
# Model: StudentModel	 Epoch: 40	 Loss: 18.9822	 Accuracy: 0.9299
# Model: StudentModel	 Epoch: 41	 Loss: 18.9768	 Accuracy: 0.9305
# Model: StudentModel	 Epoch: 42	 Loss: 18.9698	 Accuracy: 0.9287
# Model: StudentModel	 Epoch: 43	 Loss: 18.9606	 Accuracy: 0.9279
# Model: StudentModel	 Epoch: 44	 Loss: 18.9529	 Accuracy: 0.9282
# Model: StudentModel	 Epoch: 45	 Loss: 18.9453	 Accuracy: 0.9318
# Model: StudentModel	 Epoch: 46	 Loss: 18.9405	 Accuracy: 0.9297
# Model: StudentModel	 Epoch: 47	 Loss: 18.9327	 Accuracy: 0.9315
# Model: StudentModel	 Epoch: 48	 Loss: 18.9258	 Accuracy: 0.9309
# Model: StudentModel	 Epoch: 49	 Loss: 18.9202	 Accuracy: 0.9321
# Model: StudentModel	 Epoch: 50	 Loss: 18.9167	 Accuracy: 0.9321
# Model 4 name: StudentModelC
# 100%|██████████| 1500/1500 [08:38<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 1	 Loss: 0.6132	 Accuracy: 0.7397
# 100%|██████████| 1500/1500 [08:36<00:00,  2.90it/s]
# Model: Kmeans 	 Epoch: 2	 Loss: 0.5541	 Accuracy: 0.7736
# 100%|██████████| 1500/1500 [08:38<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 3	 Loss: 0.5446	 Accuracy: 0.8016
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 4	 Loss: 0.5377	 Accuracy: 0.8260
# 100%|██████████| 1500/1500 [08:38<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 5	 Loss: 0.5306	 Accuracy: 0.8385
# 100%|██████████| 1500/1500 [08:35<00:00,  2.91it/s]
# Model: Kmeans 	 Epoch: 6	 Loss: 0.5249	 Accuracy: 0.8477
# 100%|██████████| 1500/1500 [08:35<00:00,  2.91it/s]
# Model: Kmeans 	 Epoch: 7	 Loss: 0.5204	 Accuracy: 0.8570
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
# Model: Kmeans 	 Epoch: 8	 Loss: 0.5162	 Accuracy: 0.8623
# 100%|██████████| 1500/1500 [08:36<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 9	 Loss: 0.5132	 Accuracy: 0.8651
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 10	 Loss: 0.5111	 Accuracy: 0.8704
# 100%|██████████| 1500/1500 [08:38<00:00,  2.89it/s]
# Model: Kmeans 	 Epoch: 11	 Loss: 0.5094	 Accuracy: 0.8699
# 100%|██████████| 1500/1500 [08:38<00:00,  2.89it/s]
# Model: Kmeans 	 Epoch: 12	 Loss: 0.5083	 Accuracy: 0.8749
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
# Model: Kmeans 	 Epoch: 13	 Loss: 0.5071	 Accuracy: 0.8668
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 14	 Loss: 0.5062	 Accuracy: 0.8724
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
# Model: Kmeans 	 Epoch: 15	 Loss: 0.5054	 Accuracy: 0.8769
# 100%|██████████| 1500/1500 [08:33<00:00,  2.92it/s]
# Model: Kmeans 	 Epoch: 16	 Loss: 0.5046	 Accuracy: 0.8770
# 100%|██████████| 1500/1500 [08:35<00:00,  2.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 17	 Loss: 0.5042	 Accuracy: 0.8801
# 100%|██████████| 1500/1500 [08:36<00:00,  2.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 18	 Loss: 0.5033	 Accuracy: 0.8776
# 100%|██████████| 1500/1500 [08:36<00:00,  2.90it/s]
# Model: Kmeans 	 Epoch: 19	 Loss: 0.5030	 Accuracy: 0.8755
# 100%|██████████| 1500/1500 [08:33<00:00,  2.92it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 20	 Loss: 0.5025	 Accuracy: 0.8865
# 100%|██████████| 1500/1500 [08:39<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 21	 Loss: 0.5020	 Accuracy: 0.8806
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 22	 Loss: 0.5015	 Accuracy: 0.8848
# 100%|██████████| 1500/1500 [08:38<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 23	 Loss: 0.5010	 Accuracy: 0.8839
# 100%|██████████| 1500/1500 [08:37<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 24	 Loss: 0.5007	 Accuracy: 0.8865
# 100%|██████████| 1500/1500 [08:38<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 25	 Loss: 0.5002	 Accuracy: 0.8792
# 100%|██████████| 1500/1500 [08:38<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 26	 Loss: 0.4998	 Accuracy: 0.8909
# 100%|██████████| 1500/1500 [08:40<00:00,  2.88it/s]
# Model: Kmeans 	 Epoch: 27	 Loss: 0.4995	 Accuracy: 0.8850
# 100%|██████████| 1500/1500 [08:41<00:00,  2.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 28	 Loss: 0.4992	 Accuracy: 0.8890
# 100%|██████████| 1500/1500 [08:41<00:00,  2.87it/s]
# Model: Kmeans 	 Epoch: 29	 Loss: 0.4989	 Accuracy: 0.8900
# 100%|██████████| 1500/1500 [08:40<00:00,  2.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 30	 Loss: 0.4986	 Accuracy: 0.8847
# 100%|██████████| 1500/1500 [08:38<00:00,  2.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 31	 Loss: 0.4983	 Accuracy: 0.8886
# 100%|██████████| 1500/1500 [08:44<00:00,  2.86it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 32	 Loss: 0.4979	 Accuracy: 0.8912
# 100%|██████████| 1500/1500 [09:21<00:00,  2.67it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 33	 Loss: 0.4975	 Accuracy: 0.8930
# 100%|██████████| 1500/1500 [10:19<00:00,  2.42it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 34	 Loss: 0.4973	 Accuracy: 0.8922
# 100%|██████████| 1500/1500 [10:46<00:00,  2.32it/s]
# Model: Kmeans 	 Epoch: 35	 Loss: 0.4969	 Accuracy: 0.8928
# 100%|██████████| 1500/1500 [10:02<00:00,  2.49it/s]
# Model: Kmeans 	 Epoch: 36	 Loss: 0.4966	 Accuracy: 0.8932
# 100%|██████████| 1500/1500 [13:46<00:00,  1.82it/s]
# Model: Kmeans 	 Epoch: 37	 Loss: 0.4963	 Accuracy: 0.8950
# 100%|██████████| 1500/1500 [13:46<00:00,  1.81it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 38	 Loss: 0.4959	 Accuracy: 0.8924
# 100%|██████████| 1500/1500 [14:15<00:00,  1.75it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 39	 Loss: 0.4956	 Accuracy: 0.8929
# 100%|██████████| 1500/1500 [12:58<00:00,  1.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 40	 Loss: 0.4953	 Accuracy: 0.8973
# 100%|██████████| 1500/1500 [11:38<00:00,  2.15it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 41	 Loss: 0.4949	 Accuracy: 0.8911
# 100%|██████████| 1500/1500 [09:09<00:00,  2.73it/s]
# Model: Kmeans 	 Epoch: 42	 Loss: 0.4944	 Accuracy: 0.8978
# 100%|██████████| 1500/1500 [09:22<00:00,  2.67it/s]
# Model: Kmeans 	 Epoch: 43	 Loss: 0.4942	 Accuracy: 0.8934
# 100%|██████████| 1500/1500 [09:19<00:00,  2.68it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 44	 Loss: 0.4939	 Accuracy: 0.8960
# 100%|██████████| 1500/1500 [10:17<00:00,  2.43it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 45	 Loss: 0.4937	 Accuracy: 0.8965
# 100%|██████████| 1500/1500 [09:37<00:00,  2.60it/s]
# Model: Kmeans 	 Epoch: 46	 Loss: 0.4933	 Accuracy: 0.8967
# 100%|██████████| 1500/1500 [09:07<00:00,  2.74it/s]
# Model: Kmeans 	 Epoch: 47	 Loss: 0.4930	 Accuracy: 0.8938
# 100%|██████████| 1500/1500 [08:53<00:00,  2.81it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 48	 Loss: 0.4928	 Accuracy: 0.8982
# 100%|██████████| 1500/1500 [08:42<00:00,  2.87it/s]
# Model: Kmeans 	 Epoch: 49	 Loss: 0.4926	 Accuracy: 0.9008
# 100%|██████████| 1500/1500 [08:42<00:00,  2.87it/s]
# Model: Kmeans 	 Epoch: 50	 Loss: 0.4923	 Accuracy: 0.8999
# Save plot as ./alpha_value/alpha_value_0.30000000000000004.png
# Model 1 name: TeacherModelCNN
# Model: TeacherModel	 Epoch: 1	 Loss: 0.4975	 Accuracy: 0.9213
# Model: TeacherModel	 Epoch: 2	 Loss: 0.2613	 Accuracy: 0.9440
# Model: TeacherModel	 Epoch: 3	 Loss: 0.2074	 Accuracy: 0.9528
# Model: TeacherModel	 Epoch: 4	 Loss: 0.1810	 Accuracy: 0.9595
# Model: TeacherModel	 Epoch: 5	 Loss: 0.1591	 Accuracy: 0.9623
# Model: TeacherModel	 Epoch: 6	 Loss: 0.1479	 Accuracy: 0.9684
# Model: TeacherModel	 Epoch: 7	 Loss: 0.1389	 Accuracy: 0.9657
# Model: TeacherModel	 Epoch: 8	 Loss: 0.1280	 Accuracy: 0.9673
# Model: TeacherModel	 Epoch: 9	 Loss: 0.1227	 Accuracy: 0.9717
# Model: TeacherModel	 Epoch: 10	 Loss: 0.1150	 Accuracy: 0.9688
# Model: TeacherModel	 Epoch: 11	 Loss: 0.1110	 Accuracy: 0.9732
# Model: TeacherModel	 Epoch: 12	 Loss: 0.1072	 Accuracy: 0.9752
# Model: TeacherModel	 Epoch: 13	 Loss: 0.1016	 Accuracy: 0.9738
# Model: TeacherModel	 Epoch: 14	 Loss: 0.0962	 Accuracy: 0.9735
# Model: TeacherModel	 Epoch: 15	 Loss: 0.0923	 Accuracy: 0.9739
# Model: TeacherModel	 Epoch: 16	 Loss: 0.0918	 Accuracy: 0.9775
# Model: TeacherModel	 Epoch: 17	 Loss: 0.0895	 Accuracy: 0.9775
# Model: TeacherModel	 Epoch: 18	 Loss: 0.0827	 Accuracy: 0.9763
# Model: TeacherModel	 Epoch: 19	 Loss: 0.0864	 Accuracy: 0.9790
# Model: TeacherModel	 Epoch: 20	 Loss: 0.0817	 Accuracy: 0.9764
# Model: TeacherModel	 Epoch: 21	 Loss: 0.0787	 Accuracy: 0.9777
# Model: TeacherModel	 Epoch: 22	 Loss: 0.0757	 Accuracy: 0.9781
# Model: TeacherModel	 Epoch: 23	 Loss: 0.0728	 Accuracy: 0.9795
# Model: TeacherModel	 Epoch: 24	 Loss: 0.0721	 Accuracy: 0.9778
# Model: TeacherModel	 Epoch: 25	 Loss: 0.0726	 Accuracy: 0.9789
# Model: TeacherModel	 Epoch: 26	 Loss: 0.0700	 Accuracy: 0.9781
# Model: TeacherModel	 Epoch: 27	 Loss: 0.0672	 Accuracy: 0.9819
# Model: TeacherModel	 Epoch: 28	 Loss: 0.0641	 Accuracy: 0.9794
# Model: TeacherModel	 Epoch: 29	 Loss: 0.0666	 Accuracy: 0.9804
# Model: TeacherModel	 Epoch: 30	 Loss: 0.0646	 Accuracy: 0.9791
# Model: TeacherModel	 Epoch: 31	 Loss: 0.0610	 Accuracy: 0.9792
# Model: TeacherModel	 Epoch: 32	 Loss: 0.0616	 Accuracy: 0.9807
# Model: TeacherModel	 Epoch: 33	 Loss: 0.0607	 Accuracy: 0.9797
# Model: TeacherModel	 Epoch: 34	 Loss: 0.0585	 Accuracy: 0.9806
# Model: TeacherModel	 Epoch: 35	 Loss: 0.0579	 Accuracy: 0.9798
# Model: TeacherModel	 Epoch: 36	 Loss: 0.0585	 Accuracy: 0.9800
# Model: TeacherModel	 Epoch: 37	 Loss: 0.0579	 Accuracy: 0.9818
# Model: TeacherModel	 Epoch: 38	 Loss: 0.0545	 Accuracy: 0.9792
# Model: TeacherModel	 Epoch: 39	 Loss: 0.0553	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 40	 Loss: 0.0555	 Accuracy: 0.9828
# Model: TeacherModel	 Epoch: 41	 Loss: 0.0532	 Accuracy: 0.9822
# Model: TeacherModel	 Epoch: 42	 Loss: 0.0515	 Accuracy: 0.9819
# Model: TeacherModel	 Epoch: 43	 Loss: 0.0519	 Accuracy: 0.9825
# Model: TeacherModel	 Epoch: 44	 Loss: 0.0540	 Accuracy: 0.9816
# Model: TeacherModel	 Epoch: 45	 Loss: 0.0504	 Accuracy: 0.9830
# Model: TeacherModel	 Epoch: 46	 Loss: 0.0512	 Accuracy: 0.9838
# Model: TeacherModel	 Epoch: 47	 Loss: 0.0500	 Accuracy: 0.9821
# Model: TeacherModel	 Epoch: 48	 Loss: 0.0459	 Accuracy: 0.9848
# Model: TeacherModel	 Epoch: 49	 Loss: 0.0455	 Accuracy: 0.9815
# Model: TeacherModel	 Epoch: 50	 Loss: 0.0496	 Accuracy: 0.9847
# Model 2 name: StudentModelA
# Model: StudentModel	 Epoch: 1	 Loss: 0.9169	 Accuracy: 0.8597
# Model: StudentModel	 Epoch: 2	 Loss: 0.4335	 Accuracy: 0.8867
# Model: StudentModel	 Epoch: 3	 Loss: 0.3662	 Accuracy: 0.8988
# Model: StudentModel	 Epoch: 4	 Loss: 0.3368	 Accuracy: 0.8992
# Model: StudentModel	 Epoch: 5	 Loss: 0.3186	 Accuracy: 0.9041
# Model: StudentModel	 Epoch: 6	 Loss: 0.3058	 Accuracy: 0.9086
# Model: StudentModel	 Epoch: 7	 Loss: 0.2947	 Accuracy: 0.9128
# Model: StudentModel	 Epoch: 8	 Loss: 0.2863	 Accuracy: 0.9145
# Model: StudentModel	 Epoch: 9	 Loss: 0.2771	 Accuracy: 0.9181
# Model: StudentModel	 Epoch: 10	 Loss: 0.2689	 Accuracy: 0.9177
# Model: StudentModel	 Epoch: 11	 Loss: 0.2607	 Accuracy: 0.9200
# Model: StudentModel	 Epoch: 12	 Loss: 0.2548	 Accuracy: 0.9248
# Model: StudentModel	 Epoch: 13	 Loss: 0.2473	 Accuracy: 0.9229
# Model: StudentModel	 Epoch: 14	 Loss: 0.2416	 Accuracy: 0.9237
# Model: StudentModel	 Epoch: 15	 Loss: 0.2364	 Accuracy: 0.9249
# Model: StudentModel	 Epoch: 16	 Loss: 0.2314	 Accuracy: 0.9270
# Model: StudentModel	 Epoch: 17	 Loss: 0.2273	 Accuracy: 0.9278
# Model: StudentModel	 Epoch: 18	 Loss: 0.2236	 Accuracy: 0.9294
# Model: StudentModel	 Epoch: 19	 Loss: 0.2204	 Accuracy: 0.9312
# Model: StudentModel	 Epoch: 20	 Loss: 0.2174	 Accuracy: 0.9293
# Model: StudentModel	 Epoch: 21	 Loss: 0.2144	 Accuracy: 0.9322
# Model: StudentModel	 Epoch: 22	 Loss: 0.2114	 Accuracy: 0.9339
# Model: StudentModel	 Epoch: 23	 Loss: 0.2091	 Accuracy: 0.9327
# Model: StudentModel	 Epoch: 24	 Loss: 0.2062	 Accuracy: 0.9311
# Model: StudentModel	 Epoch: 25	 Loss: 0.2048	 Accuracy: 0.9326
# Model: StudentModel	 Epoch: 26	 Loss: 0.2027	 Accuracy: 0.9343
# Model: StudentModel	 Epoch: 27	 Loss: 0.1996	 Accuracy: 0.9322
# Model: StudentModel	 Epoch: 28	 Loss: 0.1987	 Accuracy: 0.9347
# Model: StudentModel	 Epoch: 29	 Loss: 0.1959	 Accuracy: 0.9353
# Model: StudentModel	 Epoch: 30	 Loss: 0.1950	 Accuracy: 0.9352
# Model: StudentModel	 Epoch: 31	 Loss: 0.1924	 Accuracy: 0.9332
# Model: StudentModel	 Epoch: 32	 Loss: 0.1913	 Accuracy: 0.9322
# Model: StudentModel	 Epoch: 33	 Loss: 0.1899	 Accuracy: 0.9346
# Model: StudentModel	 Epoch: 34	 Loss: 0.1875	 Accuracy: 0.9348
# Model: StudentModel	 Epoch: 35	 Loss: 0.1868	 Accuracy: 0.9332
# Model: StudentModel	 Epoch: 36	 Loss: 0.1856	 Accuracy: 0.9349
# Model: StudentModel	 Epoch: 37	 Loss: 0.1838	 Accuracy: 0.9382
# Model: StudentModel	 Epoch: 38	 Loss: 0.1826	 Accuracy: 0.9360
# Model: StudentModel	 Epoch: 39	 Loss: 0.1817	 Accuracy: 0.9356
# Model: StudentModel	 Epoch: 40	 Loss: 0.1802	 Accuracy: 0.9377
# Model: StudentModel	 Epoch: 41	 Loss: 0.1795	 Accuracy: 0.9354
# Model: StudentModel	 Epoch: 42	 Loss: 0.1785	 Accuracy: 0.9351
# Model: StudentModel	 Epoch: 43	 Loss: 0.1778	 Accuracy: 0.9374
# Model: StudentModel	 Epoch: 44	 Loss: 0.1758	 Accuracy: 0.9377
# Model: StudentModel	 Epoch: 45	 Loss: 0.1751	 Accuracy: 0.9370
# Model: StudentModel	 Epoch: 46	 Loss: 0.1744	 Accuracy: 0.9374
# Model: StudentModel	 Epoch: 47	 Loss: 0.1731	 Accuracy: 0.9377
# Model: StudentModel	 Epoch: 48	 Loss: 0.1728	 Accuracy: 0.9378
# Model: StudentModel	 Epoch: 49	 Loss: 0.1707	 Accuracy: 0.9375
# Model: StudentModel	 Epoch: 50	 Loss: 0.1705	 Accuracy: 0.9356
# Model 3 name: StudentModelB
# Model: StudentModel	 Epoch: 1	 Loss: 25.1811	 Accuracy: 0.8367
# Model: StudentModel	 Epoch: 2	 Loss: 20.7076	 Accuracy: 0.8841
# Model: StudentModel	 Epoch: 3	 Loss: 20.1162	 Accuracy: 0.8914
# Model: StudentModel	 Epoch: 4	 Loss: 19.8978	 Accuracy: 0.8989
# Model: StudentModel	 Epoch: 5	 Loss: 19.7577	 Accuracy: 0.9037
# Model: StudentModel	 Epoch: 6	 Loss: 19.6494	 Accuracy: 0.9091
# Model: StudentModel	 Epoch: 7	 Loss: 19.5647	 Accuracy: 0.9126
# Model: StudentModel	 Epoch: 8	 Loss: 19.4909	 Accuracy: 0.9135
# Model: StudentModel	 Epoch: 9	 Loss: 19.4261	 Accuracy: 0.9174
# Model: StudentModel	 Epoch: 10	 Loss: 19.3689	 Accuracy: 0.9197
# Model: StudentModel	 Epoch: 11	 Loss: 19.3157	 Accuracy: 0.9198
# Model: StudentModel	 Epoch: 12	 Loss: 19.2694	 Accuracy: 0.9197
# Model: StudentModel	 Epoch: 13	 Loss: 19.2287	 Accuracy: 0.9227
# Model: StudentModel	 Epoch: 14	 Loss: 19.1898	 Accuracy: 0.9257
# Model: StudentModel	 Epoch: 15	 Loss: 19.1567	 Accuracy: 0.9263
# Model: StudentModel	 Epoch: 16	 Loss: 19.1254	 Accuracy: 0.9244
# Model: StudentModel	 Epoch: 17	 Loss: 19.0997	 Accuracy: 0.9267
# Model: StudentModel	 Epoch: 18	 Loss: 19.0708	 Accuracy: 0.9290
# Model: StudentModel	 Epoch: 19	 Loss: 19.0489	 Accuracy: 0.9293
# Model: StudentModel	 Epoch: 20	 Loss: 19.0252	 Accuracy: 0.9290
# Model: StudentModel	 Epoch: 21	 Loss: 19.0042	 Accuracy: 0.9326
# Model: StudentModel	 Epoch: 22	 Loss: 18.9825	 Accuracy: 0.9316
# Model: StudentModel	 Epoch: 23	 Loss: 18.9595	 Accuracy: 0.9312
# Model: StudentModel	 Epoch: 24	 Loss: 18.9432	 Accuracy: 0.9339
# Model: StudentModel	 Epoch: 25	 Loss: 18.9227	 Accuracy: 0.9352
# Model: StudentModel	 Epoch: 26	 Loss: 18.9023	 Accuracy: 0.9344
# Model: StudentModel	 Epoch: 27	 Loss: 18.8894	 Accuracy: 0.9360
# Model: StudentModel	 Epoch: 28	 Loss: 18.8720	 Accuracy: 0.9350
# Model: StudentModel	 Epoch: 29	 Loss: 18.8591	 Accuracy: 0.9368
# Model: StudentModel	 Epoch: 30	 Loss: 18.8443	 Accuracy: 0.9352
# Model: StudentModel	 Epoch: 31	 Loss: 18.8326	 Accuracy: 0.9377
# Model: StudentModel	 Epoch: 32	 Loss: 18.8202	 Accuracy: 0.9379
# Model: StudentModel	 Epoch: 33	 Loss: 18.8075	 Accuracy: 0.9348
# Model: StudentModel	 Epoch: 34	 Loss: 18.7980	 Accuracy: 0.9386
# Model: StudentModel	 Epoch: 35	 Loss: 18.7853	 Accuracy: 0.9383
# Model: StudentModel	 Epoch: 36	 Loss: 18.7775	 Accuracy: 0.9387
# Model: StudentModel	 Epoch: 37	 Loss: 18.7682	 Accuracy: 0.9381
# Model: StudentModel	 Epoch: 38	 Loss: 18.7561	 Accuracy: 0.9394
# Model: StudentModel	 Epoch: 39	 Loss: 18.7499	 Accuracy: 0.9397
# Model: StudentModel	 Epoch: 40	 Loss: 18.7383	 Accuracy: 0.9393
# Model: StudentModel	 Epoch: 41	 Loss: 18.7314	 Accuracy: 0.9395
# Model: StudentModel	 Epoch: 42	 Loss: 18.7254	 Accuracy: 0.9403
# Model: StudentModel	 Epoch: 43	 Loss: 18.7174	 Accuracy: 0.9402
# Model: StudentModel	 Epoch: 44	 Loss: 18.7095	 Accuracy: 0.9396
# Model: StudentModel	 Epoch: 45	 Loss: 18.6999	 Accuracy: 0.9376
# Model: StudentModel	 Epoch: 46	 Loss: 18.6923	 Accuracy: 0.9426
# Model: StudentModel	 Epoch: 47	 Loss: 18.6851	 Accuracy: 0.9429
# Model: StudentModel	 Epoch: 48	 Loss: 18.6781	 Accuracy: 0.9412
# Model: StudentModel	 Epoch: 49	 Loss: 18.6717	 Accuracy: 0.9414
# Model: StudentModel	 Epoch: 50	 Loss: 18.6641	 Accuracy: 0.9427
# Model 4 name: StudentModelC
# 100%|██████████| 1500/1500 [08:47<00:00,  2.85it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 1	 Loss: 0.4847	 Accuracy: 0.5342
# 100%|██████████| 1500/1500 [09:23<00:00,  2.66it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 2	 Loss: 0.4421	 Accuracy: 0.5657
# 100%|██████████| 1500/1500 [09:49<00:00,  2.54it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 3	 Loss: 0.4348	 Accuracy: 0.5883
# 100%|██████████| 1500/1500 [09:58<00:00,  2.50it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 4	 Loss: 0.4303	 Accuracy: 0.6042
# 100%|██████████| 1500/1500 [10:28<00:00,  2.39it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 5	 Loss: 0.4273	 Accuracy: 0.6092
# 100%|██████████| 1500/1500 [11:57<00:00,  2.09it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 6	 Loss: 0.4244	 Accuracy: 0.6214
# 100%|██████████| 1500/1500 [11:33<00:00,  2.16it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 7	 Loss: 0.4219	 Accuracy: 0.6373
# 100%|██████████| 1500/1500 [10:14<00:00,  2.44it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 8	 Loss: 0.4198	 Accuracy: 0.6466
# 100%|██████████| 1500/1500 [09:53<00:00,  2.53it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 9	 Loss: 0.4180	 Accuracy: 0.6704
# 100%|██████████| 1500/1500 [11:03<00:00,  2.26it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 10	 Loss: 0.4170	 Accuracy: 0.6764
# 100%|██████████| 1500/1500 [10:24<00:00,  2.40it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 11	 Loss: 0.4162	 Accuracy: 0.6787
# 100%|██████████| 1500/1500 [10:00<00:00,  2.50it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 12	 Loss: 0.4156	 Accuracy: 0.6636
# 100%|██████████| 1500/1500 [08:53<00:00,  2.81it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 13	 Loss: 0.4150	 Accuracy: 0.6824
# 100%|██████████| 1500/1500 [08:45<00:00,  2.86it/s]
# Model: Kmeans 	 Epoch: 14	 Loss: 0.4145	 Accuracy: 0.6959
# 100%|██████████| 1500/1500 [08:53<00:00,  2.81it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 15	 Loss: 0.4141	 Accuracy: 0.6951
# 100%|██████████| 1500/1500 [09:10<00:00,  2.72it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 16	 Loss: 0.4138	 Accuracy: 0.6980
# 100%|██████████| 1500/1500 [09:13<00:00,  2.71it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 17	 Loss: 0.4135	 Accuracy: 0.7141
# 100%|██████████| 1500/1500 [10:11<00:00,  2.45it/s]
# Model: Kmeans 	 Epoch: 18	 Loss: 0.4132	 Accuracy: 0.7045
# 100%|██████████| 1500/1500 [10:51<00:00,  2.30it/s]
# Model: Kmeans 	 Epoch: 19	 Loss: 0.4128	 Accuracy: 0.6965
# 100%|██████████| 1500/1500 [09:34<00:00,  2.61it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 20	 Loss: 0.4122	 Accuracy: 0.7060
# 100%|██████████| 1500/1500 [09:49<00:00,  2.54it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 21	 Loss: 0.4116	 Accuracy: 0.7057
# 100%|██████████| 1500/1500 [09:08<00:00,  2.73it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 22	 Loss: 0.4112	 Accuracy: 0.7331
# 100%|██████████| 1500/1500 [09:12<00:00,  2.71it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 23	 Loss: 0.4107	 Accuracy: 0.7255
# 100%|██████████| 1500/1500 [09:14<00:00,  2.70it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 24	 Loss: 0.4102	 Accuracy: 0.7083
# 100%|██████████| 1500/1500 [09:07<00:00,  2.74it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 25	 Loss: 0.4098	 Accuracy: 0.7164
# 100%|██████████| 1500/1500 [10:56<00:00,  2.29it/s]
# Model: Kmeans 	 Epoch: 26	 Loss: 0.4094	 Accuracy: 0.7139
# 100%|██████████| 1500/1500 [13:03<00:00,  1.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 27	 Loss: 0.4091	 Accuracy: 0.7069
# 100%|██████████| 1500/1500 [20:21<00:00,  1.23it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 28	 Loss: 0.4087	 Accuracy: 0.7398
# 100%|██████████| 1500/1500 [28:34<00:00,  1.14s/it]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 29	 Loss: 0.4083	 Accuracy: 0.7368
# 100%|██████████| 1500/1500 [12:55<00:00,  1.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 30	 Loss: 0.4080	 Accuracy: 0.7367
# 100%|██████████| 1500/1500 [26:22<00:00,  1.05s/it]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 31	 Loss: 0.4076	 Accuracy: 0.7243
# 100%|██████████| 1500/1500 [23:20<00:00,  1.07it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 32	 Loss: 0.4072	 Accuracy: 0.7369
# 100%|██████████| 1500/1500 [13:09<00:00,  1.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 33	 Loss: 0.4069	 Accuracy: 0.7238
# 100%|██████████| 1500/1500 [12:39<00:00,  1.97it/s]
# Model: Kmeans 	 Epoch: 34	 Loss: 0.4067	 Accuracy: 0.7396
# 100%|██████████| 1500/1500 [13:09<00:00,  1.90it/s]
# Model: Kmeans 	 Epoch: 35	 Loss: 0.4063	 Accuracy: 0.7258
# 100%|██████████| 1500/1500 [13:05<00:00,  1.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 36	 Loss: 0.4061	 Accuracy: 0.7339
# 100%|██████████| 1500/1500 [13:15<00:00,  1.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 37	 Loss: 0.4059	 Accuracy: 0.7303
# 100%|██████████| 1500/1500 [25:26<00:00,  1.02s/it]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 38	 Loss: 0.4057	 Accuracy: 0.7312
# 100%|██████████| 1500/1500 [27:32<00:00,  1.10s/it]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 39	 Loss: 0.4054	 Accuracy: 0.7310
# 100%|██████████| 1500/1500 [22:23<00:00,  1.12it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 40	 Loss: 0.4053	 Accuracy: 0.7317
# 100%|██████████| 1500/1500 [26:33<00:00,  1.06s/it]
# Model: Kmeans 	 Epoch: 41	 Loss: 0.4051	 Accuracy: 0.7473
# 100%|██████████| 1500/1500 [26:46<00:00,  1.07s/it]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 42	 Loss: 0.4049	 Accuracy: 0.7513
# 100%|██████████| 1500/1500 [12:55<00:00,  1.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 43	 Loss: 0.4047	 Accuracy: 0.7458
# 100%|██████████| 1500/1500 [12:45<00:00,  1.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 44	 Loss: 0.4045	 Accuracy: 0.7288
# 100%|██████████| 1500/1500 [13:03<00:00,  1.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 45	 Loss: 0.4043	 Accuracy: 0.7368
# 100%|██████████| 1500/1500 [13:27<00:00,  1.86it/s]
# Model: Kmeans 	 Epoch: 46	 Loss: 0.4041	 Accuracy: 0.7390
# 100%|██████████| 1500/1500 [13:05<00:00,  1.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 47	 Loss: 0.4039	 Accuracy: 0.7424
# 100%|██████████| 1500/1500 [12:54<00:00,  1.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 48	 Loss: 0.4038	 Accuracy: 0.7433
# 100%|██████████| 1500/1500 [12:56<00:00,  1.93it/s]
# Model: Kmeans 	 Epoch: 49	 Loss: 0.4036	 Accuracy: 0.7551
# 100%|██████████| 1500/1500 [12:52<00:00,  1.94it/s]
# Model: Kmeans 	 Epoch: 50	 Loss: 0.4034	 Accuracy: 0.7443
# Save plot as ./alpha_value/alpha_value_0.19999999999999996.png
# Model 1 name: TeacherModelCNN
# Model: TeacherModel	 Epoch: 1	 Loss: 0.4974	 Accuracy: 0.9217
# Model: TeacherModel	 Epoch: 2	 Loss: 0.2597	 Accuracy: 0.9470
# Model: TeacherModel	 Epoch: 3	 Loss: 0.2052	 Accuracy: 0.9557
# Model: TeacherModel	 Epoch: 4	 Loss: 0.1765	 Accuracy: 0.9603
# Model: TeacherModel	 Epoch: 5	 Loss: 0.1580	 Accuracy: 0.9628
# Model: TeacherModel	 Epoch: 6	 Loss: 0.1451	 Accuracy: 0.9641
# Model: TeacherModel	 Epoch: 7	 Loss: 0.1329	 Accuracy: 0.9695
# Model: TeacherModel	 Epoch: 8	 Loss: 0.1290	 Accuracy: 0.9703
# Model: TeacherModel	 Epoch: 9	 Loss: 0.1183	 Accuracy: 0.9702
# Model: TeacherModel	 Epoch: 10	 Loss: 0.1136	 Accuracy: 0.9688
# Model: TeacherModel	 Epoch: 11	 Loss: 0.1090	 Accuracy: 0.9730
# Model: TeacherModel	 Epoch: 12	 Loss: 0.1037	 Accuracy: 0.9716
# Model: TeacherModel	 Epoch: 13	 Loss: 0.1009	 Accuracy: 0.9735
# Model: TeacherModel	 Epoch: 14	 Loss: 0.0985	 Accuracy: 0.9745
# Model: TeacherModel	 Epoch: 15	 Loss: 0.0909	 Accuracy: 0.9741
# Model: TeacherModel	 Epoch: 16	 Loss: 0.0905	 Accuracy: 0.9755
# Model: TeacherModel	 Epoch: 17	 Loss: 0.0825	 Accuracy: 0.9743
# Model: TeacherModel	 Epoch: 18	 Loss: 0.0853	 Accuracy: 0.9788
# Model: TeacherModel	 Epoch: 19	 Loss: 0.0821	 Accuracy: 0.9773
# Model: TeacherModel	 Epoch: 20	 Loss: 0.0790	 Accuracy: 0.9782
# Model: TeacherModel	 Epoch: 21	 Loss: 0.0753	 Accuracy: 0.9778
# Model: TeacherModel	 Epoch: 22	 Loss: 0.0748	 Accuracy: 0.9765
# Model: TeacherModel	 Epoch: 23	 Loss: 0.0725	 Accuracy: 0.9783
# Model: TeacherModel	 Epoch: 24	 Loss: 0.0729	 Accuracy: 0.9769
# Model: TeacherModel	 Epoch: 25	 Loss: 0.0719	 Accuracy: 0.9763
# Model: TeacherModel	 Epoch: 26	 Loss: 0.0700	 Accuracy: 0.9802
# Model: TeacherModel	 Epoch: 27	 Loss: 0.0662	 Accuracy: 0.9804
# Model: TeacherModel	 Epoch: 28	 Loss: 0.0678	 Accuracy: 0.9812
# Model: TeacherModel	 Epoch: 29	 Loss: 0.0647	 Accuracy: 0.9796
# Model: TeacherModel	 Epoch: 30	 Loss: 0.0621	 Accuracy: 0.9802
# Model: TeacherModel	 Epoch: 31	 Loss: 0.0638	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 32	 Loss: 0.0604	 Accuracy: 0.9788
# Model: TeacherModel	 Epoch: 33	 Loss: 0.0574	 Accuracy: 0.9809
# Model: TeacherModel	 Epoch: 34	 Loss: 0.0619	 Accuracy: 0.9822
# Model: TeacherModel	 Epoch: 35	 Loss: 0.0574	 Accuracy: 0.9806
# Model: TeacherModel	 Epoch: 36	 Loss: 0.0588	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 37	 Loss: 0.0574	 Accuracy: 0.9784
# Model: TeacherModel	 Epoch: 38	 Loss: 0.0530	 Accuracy: 0.9798
# Model: TeacherModel	 Epoch: 39	 Loss: 0.0545	 Accuracy: 0.9807
# Model: TeacherModel	 Epoch: 40	 Loss: 0.0540	 Accuracy: 0.9770
# Model: TeacherModel	 Epoch: 41	 Loss: 0.0546	 Accuracy: 0.9828
# Model: TeacherModel	 Epoch: 42	 Loss: 0.0510	 Accuracy: 0.9816
# Model: TeacherModel	 Epoch: 43	 Loss: 0.0543	 Accuracy: 0.9813
# Model: TeacherModel	 Epoch: 44	 Loss: 0.0515	 Accuracy: 0.9817
# Model: TeacherModel	 Epoch: 45	 Loss: 0.0510	 Accuracy: 0.9807
# Model: TeacherModel	 Epoch: 46	 Loss: 0.0533	 Accuracy: 0.9804
# Model: TeacherModel	 Epoch: 47	 Loss: 0.0495	 Accuracy: 0.9822
# Model: TeacherModel	 Epoch: 48	 Loss: 0.0462	 Accuracy: 0.9832
# Model: TeacherModel	 Epoch: 49	 Loss: 0.0441	 Accuracy: 0.9821
# Model: TeacherModel	 Epoch: 50	 Loss: 0.0488	 Accuracy: 0.9805
# Model 2 name: StudentModelA
# Model: StudentModel	 Epoch: 1	 Loss: 0.9618	 Accuracy: 0.8689
# Model: StudentModel	 Epoch: 2	 Loss: 0.4162	 Accuracy: 0.8876
# Model: StudentModel	 Epoch: 3	 Loss: 0.3514	 Accuracy: 0.9016
# Model: StudentModel	 Epoch: 4	 Loss: 0.3211	 Accuracy: 0.9032
# Model: StudentModel	 Epoch: 5	 Loss: 0.3013	 Accuracy: 0.9123
# Model: StudentModel	 Epoch: 6	 Loss: 0.2843	 Accuracy: 0.9146
# Model: StudentModel	 Epoch: 7	 Loss: 0.2683	 Accuracy: 0.9180
# Model: StudentModel	 Epoch: 8	 Loss: 0.2562	 Accuracy: 0.9216
# Model: StudentModel	 Epoch: 9	 Loss: 0.2457	 Accuracy: 0.9260
# Model: StudentModel	 Epoch: 10	 Loss: 0.2358	 Accuracy: 0.9263
# Model: StudentModel	 Epoch: 11	 Loss: 0.2268	 Accuracy: 0.9225
# Model: StudentModel	 Epoch: 12	 Loss: 0.2187	 Accuracy: 0.9283
# Model: StudentModel	 Epoch: 13	 Loss: 0.2115	 Accuracy: 0.9303
# Model: StudentModel	 Epoch: 14	 Loss: 0.2057	 Accuracy: 0.9315
# Model: StudentModel	 Epoch: 15	 Loss: 0.2000	 Accuracy: 0.9331
# Model: StudentModel	 Epoch: 16	 Loss: 0.1945	 Accuracy: 0.9325
# Model: StudentModel	 Epoch: 17	 Loss: 0.1913	 Accuracy: 0.9356
# Model: StudentModel	 Epoch: 18	 Loss: 0.1863	 Accuracy: 0.9367
# Model: StudentModel	 Epoch: 19	 Loss: 0.1826	 Accuracy: 0.9353
# Model: StudentModel	 Epoch: 20	 Loss: 0.1800	 Accuracy: 0.9379
# Model: StudentModel	 Epoch: 21	 Loss: 0.1756	 Accuracy: 0.9365
# Model: StudentModel	 Epoch: 22	 Loss: 0.1727	 Accuracy: 0.9397
# Model: StudentModel	 Epoch: 23	 Loss: 0.1697	 Accuracy: 0.9393
# Model: StudentModel	 Epoch: 24	 Loss: 0.1669	 Accuracy: 0.9405
# Model: StudentModel	 Epoch: 25	 Loss: 0.1649	 Accuracy: 0.9404
# Model: StudentModel	 Epoch: 26	 Loss: 0.1626	 Accuracy: 0.9414
# Model: StudentModel	 Epoch: 27	 Loss: 0.1599	 Accuracy: 0.9424
# Model: StudentModel	 Epoch: 28	 Loss: 0.1580	 Accuracy: 0.9423
# Model: StudentModel	 Epoch: 29	 Loss: 0.1556	 Accuracy: 0.9419
# Model: StudentModel	 Epoch: 30	 Loss: 0.1539	 Accuracy: 0.9447
# Model: StudentModel	 Epoch: 31	 Loss: 0.1528	 Accuracy: 0.9447
# Model: StudentModel	 Epoch: 32	 Loss: 0.1505	 Accuracy: 0.9435
# Model: StudentModel	 Epoch: 33	 Loss: 0.1489	 Accuracy: 0.9440
# Model: StudentModel	 Epoch: 34	 Loss: 0.1479	 Accuracy: 0.9440
# Model: StudentModel	 Epoch: 35	 Loss: 0.1468	 Accuracy: 0.9427
# Model: StudentModel	 Epoch: 36	 Loss: 0.1445	 Accuracy: 0.9447
# Model: StudentModel	 Epoch: 37	 Loss: 0.1434	 Accuracy: 0.9440
# Model: StudentModel	 Epoch: 38	 Loss: 0.1419	 Accuracy: 0.9458
# Model: StudentModel	 Epoch: 39	 Loss: 0.1412	 Accuracy: 0.9455
# Model: StudentModel	 Epoch: 40	 Loss: 0.1396	 Accuracy: 0.9480
# Model: StudentModel	 Epoch: 41	 Loss: 0.1388	 Accuracy: 0.9448
# Model: StudentModel	 Epoch: 42	 Loss: 0.1376	 Accuracy: 0.9490
# Model: StudentModel	 Epoch: 43	 Loss: 0.1362	 Accuracy: 0.9452
# Model: StudentModel	 Epoch: 44	 Loss: 0.1358	 Accuracy: 0.9446
# Model: StudentModel	 Epoch: 45	 Loss: 0.1343	 Accuracy: 0.9472
# Model: StudentModel	 Epoch: 46	 Loss: 0.1332	 Accuracy: 0.9462
# Model: StudentModel	 Epoch: 47	 Loss: 0.1316	 Accuracy: 0.9443
# Model: StudentModel	 Epoch: 48	 Loss: 0.1307	 Accuracy: 0.9489
# Model: StudentModel	 Epoch: 49	 Loss: 0.1303	 Accuracy: 0.9483
# Model: StudentModel	 Epoch: 50	 Loss: 0.1296	 Accuracy: 0.9484
# Model 3 name: StudentModelB
# Model: StudentModel	 Epoch: 1	 Loss: 26.1362	 Accuracy: 0.8184
# Model: StudentModel	 Epoch: 2	 Loss: 21.0562	 Accuracy: 0.8768
# Model: StudentModel	 Epoch: 3	 Loss: 20.2804	 Accuracy: 0.8871
# Model: StudentModel	 Epoch: 4	 Loss: 20.0585	 Accuracy: 0.8906
# Model: StudentModel	 Epoch: 5	 Loss: 19.9381	 Accuracy: 0.8947
# Model: StudentModel	 Epoch: 6	 Loss: 19.8563	 Accuracy: 0.8972
# Model: StudentModel	 Epoch: 7	 Loss: 19.7841	 Accuracy: 0.8999
# Model: StudentModel	 Epoch: 8	 Loss: 19.7243	 Accuracy: 0.9020
# Model: StudentModel	 Epoch: 9	 Loss: 19.6670	 Accuracy: 0.9052
# Model: StudentModel	 Epoch: 10	 Loss: 19.6176	 Accuracy: 0.9038
# Model: StudentModel	 Epoch: 11	 Loss: 19.5652	 Accuracy: 0.9094
# Model: StudentModel	 Epoch: 12	 Loss: 19.5178	 Accuracy: 0.9084
# Model: StudentModel	 Epoch: 13	 Loss: 19.4735	 Accuracy: 0.9113
# Model: StudentModel	 Epoch: 14	 Loss: 19.4337	 Accuracy: 0.9115
# Model: StudentModel	 Epoch: 15	 Loss: 19.3987	 Accuracy: 0.9144
# Model: StudentModel	 Epoch: 16	 Loss: 19.3679	 Accuracy: 0.9117
# Model: StudentModel	 Epoch: 17	 Loss: 19.3421	 Accuracy: 0.9153
# Model: StudentModel	 Epoch: 18	 Loss: 19.3142	 Accuracy: 0.9167
# Model: StudentModel	 Epoch: 19	 Loss: 19.2912	 Accuracy: 0.9183
# Model: StudentModel	 Epoch: 20	 Loss: 19.2685	 Accuracy: 0.9179
# Model: StudentModel	 Epoch: 21	 Loss: 19.2440	 Accuracy: 0.9183
# Model: StudentModel	 Epoch: 22	 Loss: 19.2240	 Accuracy: 0.9187
# Model: StudentModel	 Epoch: 23	 Loss: 19.2056	 Accuracy: 0.9185
# Model: StudentModel	 Epoch: 24	 Loss: 19.1890	 Accuracy: 0.9200
# Model: StudentModel	 Epoch: 25	 Loss: 19.1694	 Accuracy: 0.9195
# Model: StudentModel	 Epoch: 26	 Loss: 19.1507	 Accuracy: 0.9204
# Model: StudentModel	 Epoch: 27	 Loss: 19.1321	 Accuracy: 0.9203
# Model: StudentModel	 Epoch: 28	 Loss: 19.1136	 Accuracy: 0.9207
# Model: StudentModel	 Epoch: 29	 Loss: 19.0992	 Accuracy: 0.9230
# Model: StudentModel	 Epoch: 30	 Loss: 19.0821	 Accuracy: 0.9243
# Model: StudentModel	 Epoch: 31	 Loss: 19.0665	 Accuracy: 0.9243
# Model: StudentModel	 Epoch: 32	 Loss: 19.0490	 Accuracy: 0.9262
# Model: StudentModel	 Epoch: 33	 Loss: 19.0365	 Accuracy: 0.9247
# Model: StudentModel	 Epoch: 34	 Loss: 19.0205	 Accuracy: 0.9251
# Model: StudentModel	 Epoch: 35	 Loss: 19.0093	 Accuracy: 0.9272
# Model: StudentModel	 Epoch: 36	 Loss: 18.9960	 Accuracy: 0.9273
# Model: StudentModel	 Epoch: 37	 Loss: 18.9834	 Accuracy: 0.9278
# Model: StudentModel	 Epoch: 38	 Loss: 18.9736	 Accuracy: 0.9280
# Model: StudentModel	 Epoch: 39	 Loss: 18.9627	 Accuracy: 0.9292
# Model: StudentModel	 Epoch: 40	 Loss: 18.9533	 Accuracy: 0.9300
# Model: StudentModel	 Epoch: 41	 Loss: 18.9429	 Accuracy: 0.9296
# Model: StudentModel	 Epoch: 42	 Loss: 18.9316	 Accuracy: 0.9288
# Model: StudentModel	 Epoch: 43	 Loss: 18.9244	 Accuracy: 0.9302
# Model: StudentModel	 Epoch: 44	 Loss: 18.9165	 Accuracy: 0.9299
# Model: StudentModel	 Epoch: 45	 Loss: 18.9094	 Accuracy: 0.9307
# Model: StudentModel	 Epoch: 46	 Loss: 18.9000	 Accuracy: 0.9323
# Model: StudentModel	 Epoch: 47	 Loss: 18.8964	 Accuracy: 0.9302
# Model: StudentModel	 Epoch: 48	 Loss: 18.8875	 Accuracy: 0.9323
# Model: StudentModel	 Epoch: 49	 Loss: 18.8844	 Accuracy: 0.9297
# Model: StudentModel	 Epoch: 50	 Loss: 18.8728	 Accuracy: 0.9319
# Model 4 name: StudentModelC
# 100%|██████████| 1500/1500 [13:04<00:00,  1.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 1	 Loss: 0.3038	 Accuracy: 0.2542
# 100%|██████████| 1500/1500 [13:22<00:00,  1.87it/s]
# Model: Kmeans 	 Epoch: 2	 Loss: 0.2552	 Accuracy: 0.3241
# 100%|██████████| 1500/1500 [13:28<00:00,  1.86it/s]
# Model: Kmeans 	 Epoch: 3	 Loss: 0.2497	 Accuracy: 0.3337
# 100%|██████████| 1500/1500 [13:08<00:00,  1.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 4	 Loss: 0.2460	 Accuracy: 0.3444
# 100%|██████████| 1500/1500 [12:45<00:00,  1.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 5	 Loss: 0.2440	 Accuracy: 0.3534
# 100%|██████████| 1500/1500 [12:54<00:00,  1.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 6	 Loss: 0.2427	 Accuracy: 0.3254
# 100%|██████████| 1500/1500 [13:09<00:00,  1.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 7	 Loss: 0.2418	 Accuracy: 0.3645
# 100%|██████████| 1500/1500 [13:14<00:00,  1.89it/s]
# Model: Kmeans 	 Epoch: 8	 Loss: 0.2411	 Accuracy: 0.3797
# 100%|██████████| 1500/1500 [12:42<00:00,  1.97it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 9	 Loss: 0.2405	 Accuracy: 0.3832
# 100%|██████████| 1500/1500 [12:36<00:00,  1.98it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 10	 Loss: 0.2400	 Accuracy: 0.3784
# 100%|██████████| 1500/1500 [12:43<00:00,  1.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 11	 Loss: 0.2395	 Accuracy: 0.3707
# 100%|██████████| 1500/1500 [12:48<00:00,  1.95it/s]
# Model: Kmeans 	 Epoch: 12	 Loss: 0.2391	 Accuracy: 0.3924
# 100%|██████████| 1500/1500 [12:47<00:00,  1.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 13	 Loss: 0.2387	 Accuracy: 0.3895
# 100%|██████████| 1500/1500 [12:49<00:00,  1.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 14	 Loss: 0.2384	 Accuracy: 0.3921
# 100%|██████████| 1500/1500 [12:49<00:00,  1.95it/s]
# Model: Kmeans 	 Epoch: 15	 Loss: 0.2382	 Accuracy: 0.3915
# 100%|██████████| 1500/1500 [12:38<00:00,  1.98it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 16	 Loss: 0.2379	 Accuracy: 0.3973
# 100%|██████████| 1500/1500 [12:50<00:00,  1.95it/s]
# Model: Kmeans 	 Epoch: 17	 Loss: 0.2378	 Accuracy: 0.3944
# 100%|██████████| 1500/1500 [12:39<00:00,  1.98it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 18	 Loss: 0.2376	 Accuracy: 0.3900
# 100%|██████████| 1500/1500 [12:40<00:00,  1.97it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 19	 Loss: 0.2374	 Accuracy: 0.4097
# 100%|██████████| 1500/1500 [12:26<00:00,  2.01it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 20	 Loss: 0.2372	 Accuracy: 0.3962
# 100%|██████████| 1500/1500 [12:37<00:00,  1.98it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 21	 Loss: 0.2370	 Accuracy: 0.4062
# 100%|██████████| 1500/1500 [12:40<00:00,  1.97it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 22	 Loss: 0.2369	 Accuracy: 0.4062
# 100%|██████████| 1500/1500 [13:04<00:00,  1.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 23	 Loss: 0.2367	 Accuracy: 0.4049
# 100%|██████████| 1500/1500 [13:19<00:00,  1.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 24	 Loss: 0.2366	 Accuracy: 0.4083
# 100%|██████████| 1500/1500 [12:52<00:00,  1.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 25	 Loss: 0.2364	 Accuracy: 0.4050
# 100%|██████████| 1500/1500 [12:55<00:00,  1.93it/s]
# Model: Kmeans 	 Epoch: 26	 Loss: 0.2363	 Accuracy: 0.4090
# 100%|██████████| 1500/1500 [12:54<00:00,  1.94it/s]
# Model: Kmeans 	 Epoch: 27	 Loss: 0.2362	 Accuracy: 0.4064
# 100%|██████████| 1500/1500 [13:18<00:00,  1.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 28	 Loss: 0.2362	 Accuracy: 0.4052
# 100%|██████████| 1500/1500 [12:54<00:00,  1.94it/s]
# Model: Kmeans 	 Epoch: 29	 Loss: 0.2361	 Accuracy: 0.4064
# 100%|██████████| 1500/1500 [12:54<00:00,  1.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 30	 Loss: 0.2360	 Accuracy: 0.4149
# 100%|██████████| 1500/1500 [13:12<00:00,  1.89it/s]
# Model: Kmeans 	 Epoch: 31	 Loss: 0.2359	 Accuracy: 0.4060
# 100%|██████████| 1500/1500 [09:48<00:00,  2.55it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 32	 Loss: 0.2358	 Accuracy: 0.4197
# 100%|██████████| 1500/1500 [08:36<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 33	 Loss: 0.2357	 Accuracy: 0.4041
# 100%|██████████| 1500/1500 [08:40<00:00,  2.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 34	 Loss: 0.2357	 Accuracy: 0.4125
# 100%|██████████| 1500/1500 [12:17<00:00,  2.03it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 35	 Loss: 0.2356	 Accuracy: 0.4275
# 100%|██████████| 1500/1500 [13:44<00:00,  1.82it/s]
# Model: Kmeans 	 Epoch: 36	 Loss: 0.2355	 Accuracy: 0.4228
# 100%|██████████| 1500/1500 [13:04<00:00,  1.91it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 37	 Loss: 0.2355	 Accuracy: 0.4067
# 100%|██████████| 1500/1500 [13:42<00:00,  1.82it/s]
# Model: Kmeans 	 Epoch: 38	 Loss: 0.2354	 Accuracy: 0.4198
# 100%|██████████| 1500/1500 [13:14<00:00,  1.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 39	 Loss: 0.2353	 Accuracy: 0.4097
# 100%|██████████| 1500/1500 [12:46<00:00,  1.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 40	 Loss: 0.2352	 Accuracy: 0.4230
# 100%|██████████| 1500/1500 [12:41<00:00,  1.97it/s]
# Model: Kmeans 	 Epoch: 41	 Loss: 0.2352	 Accuracy: 0.4212
# 100%|██████████| 1500/1500 [12:32<00:00,  1.99it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 42	 Loss: 0.2351	 Accuracy: 0.4184
# 100%|██████████| 1500/1500 [10:48<00:00,  2.31it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 43	 Loss: 0.2351	 Accuracy: 0.4184
# 100%|██████████| 1500/1500 [13:14<00:00,  1.89it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 44	 Loss: 0.2351	 Accuracy: 0.4173
# 100%|██████████| 1500/1500 [19:41<00:00,  1.27it/s]
# Model: Kmeans 	 Epoch: 45	 Loss: 0.2351	 Accuracy: 0.4198
# 100%|██████████| 1500/1500 [28:25<00:00,  1.14s/it]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 46	 Loss: 0.2350	 Accuracy: 0.4114
# 100%|██████████| 1500/1500 [27:02<00:00,  1.08s/it]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 47	 Loss: 0.2350	 Accuracy: 0.4231
# 100%|██████████| 1500/1500 [27:39<00:00,  1.11s/it]
# Model: Kmeans 	 Epoch: 48	 Loss: 0.2349	 Accuracy: 0.4245
# 100%|██████████| 1500/1500 [17:23<00:00,  1.44it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 49	 Loss: 0.2349	 Accuracy: 0.4306
# 100%|██████████| 1500/1500 [12:13<00:00,  2.05it/s]
# Model: Kmeans 	 Epoch: 50	 Loss: 0.2349	 Accuracy: 0.4168
# Save plot as ./alpha_value/alpha_value_0.09999999999999998.png
# Model 1 name: TeacherModelCNN
# Model: TeacherModel	 Epoch: 1	 Loss: 0.4949	 Accuracy: 0.9245
# Model: TeacherModel	 Epoch: 2	 Loss: 0.2586	 Accuracy: 0.9445
# Model: TeacherModel	 Epoch: 3	 Loss: 0.2073	 Accuracy: 0.9559
# Model: TeacherModel	 Epoch: 4	 Loss: 0.1765	 Accuracy: 0.9570
# Model: TeacherModel	 Epoch: 5	 Loss: 0.1589	 Accuracy: 0.9583
# Model: TeacherModel	 Epoch: 6	 Loss: 0.1475	 Accuracy: 0.9666
# Model: TeacherModel	 Epoch: 7	 Loss: 0.1317	 Accuracy: 0.9666
# Model: TeacherModel	 Epoch: 8	 Loss: 0.1271	 Accuracy: 0.9708
# Model: TeacherModel	 Epoch: 9	 Loss: 0.1174	 Accuracy: 0.9731
# Model: TeacherModel	 Epoch: 10	 Loss: 0.1141	 Accuracy: 0.9715
# Model: TeacherModel	 Epoch: 11	 Loss: 0.1099	 Accuracy: 0.9731
# Model: TeacherModel	 Epoch: 12	 Loss: 0.1040	 Accuracy: 0.9715
# Model: TeacherModel	 Epoch: 13	 Loss: 0.0995	 Accuracy: 0.9725
# Model: TeacherModel	 Epoch: 14	 Loss: 0.0966	 Accuracy: 0.9740
# Model: TeacherModel	 Epoch: 15	 Loss: 0.0908	 Accuracy: 0.9730
# Model: TeacherModel	 Epoch: 16	 Loss: 0.0915	 Accuracy: 0.9752
# Model: TeacherModel	 Epoch: 17	 Loss: 0.0858	 Accuracy: 0.9760
# Model: TeacherModel	 Epoch: 18	 Loss: 0.0828	 Accuracy: 0.9750
# Model: TeacherModel	 Epoch: 19	 Loss: 0.0807	 Accuracy: 0.9763
# Model: TeacherModel	 Epoch: 20	 Loss: 0.0767	 Accuracy: 0.9780
# Model: TeacherModel	 Epoch: 21	 Loss: 0.0798	 Accuracy: 0.9767
# Model: TeacherModel	 Epoch: 22	 Loss: 0.0736	 Accuracy: 0.9779
# Model: TeacherModel	 Epoch: 23	 Loss: 0.0717	 Accuracy: 0.9764
# Model: TeacherModel	 Epoch: 24	 Loss: 0.0743	 Accuracy: 0.9796
# Model: TeacherModel	 Epoch: 25	 Loss: 0.0679	 Accuracy: 0.9776
# Model: TeacherModel	 Epoch: 26	 Loss: 0.0707	 Accuracy: 0.9785
# Model: TeacherModel	 Epoch: 27	 Loss: 0.0671	 Accuracy: 0.9798
# Model: TeacherModel	 Epoch: 28	 Loss: 0.0648	 Accuracy: 0.9819
# Model: TeacherModel	 Epoch: 29	 Loss: 0.0645	 Accuracy: 0.9795
# Model: TeacherModel	 Epoch: 30	 Loss: 0.0658	 Accuracy: 0.9781
# Model: TeacherModel	 Epoch: 31	 Loss: 0.0622	 Accuracy: 0.9788
# Model: TeacherModel	 Epoch: 32	 Loss: 0.0583	 Accuracy: 0.9802
# Model: TeacherModel	 Epoch: 33	 Loss: 0.0604	 Accuracy: 0.9811
# Model: TeacherModel	 Epoch: 34	 Loss: 0.0587	 Accuracy: 0.9812
# Model: TeacherModel	 Epoch: 35	 Loss: 0.0576	 Accuracy: 0.9800
# Model: TeacherModel	 Epoch: 36	 Loss: 0.0581	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 37	 Loss: 0.0534	 Accuracy: 0.9816
# Model: TeacherModel	 Epoch: 38	 Loss: 0.0552	 Accuracy: 0.9812
# Model: TeacherModel	 Epoch: 39	 Loss: 0.0533	 Accuracy: 0.9810
# Model: TeacherModel	 Epoch: 40	 Loss: 0.0530	 Accuracy: 0.9809
# Model: TeacherModel	 Epoch: 41	 Loss: 0.0534	 Accuracy: 0.9802
# Model: TeacherModel	 Epoch: 42	 Loss: 0.0503	 Accuracy: 0.9792
# Model: TeacherModel	 Epoch: 43	 Loss: 0.0529	 Accuracy: 0.9815
# Model: TeacherModel	 Epoch: 44	 Loss: 0.0524	 Accuracy: 0.9819
# Model: TeacherModel	 Epoch: 45	 Loss: 0.0483	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 46	 Loss: 0.0490	 Accuracy: 0.9813
# Model: TeacherModel	 Epoch: 47	 Loss: 0.0522	 Accuracy: 0.9822
# Model: TeacherModel	 Epoch: 48	 Loss: 0.0473	 Accuracy: 0.9808
# Model: TeacherModel	 Epoch: 49	 Loss: 0.0474	 Accuracy: 0.9814
# Model: TeacherModel	 Epoch: 50	 Loss: 0.0482	 Accuracy: 0.9808
# Model 2 name: StudentModelA
# Model: StudentModel	 Epoch: 1	 Loss: 0.8307	 Accuracy: 0.8708
# Model: StudentModel	 Epoch: 2	 Loss: 0.3964	 Accuracy: 0.8968
# Model: StudentModel	 Epoch: 3	 Loss: 0.3386	 Accuracy: 0.9054
# Model: StudentModel	 Epoch: 4	 Loss: 0.3098	 Accuracy: 0.9070
# Model: StudentModel	 Epoch: 5	 Loss: 0.2908	 Accuracy: 0.9127
# Model: StudentModel	 Epoch: 6	 Loss: 0.2765	 Accuracy: 0.9173
# Model: StudentModel	 Epoch: 7	 Loss: 0.2635	 Accuracy: 0.9183
# Model: StudentModel	 Epoch: 8	 Loss: 0.2538	 Accuracy: 0.9214
# Model: StudentModel	 Epoch: 9	 Loss: 0.2438	 Accuracy: 0.9216
# Model: StudentModel	 Epoch: 10	 Loss: 0.2356	 Accuracy: 0.9256
# Model: StudentModel	 Epoch: 11	 Loss: 0.2281	 Accuracy: 0.9225
# Model: StudentModel	 Epoch: 12	 Loss: 0.2217	 Accuracy: 0.9263
# Model: StudentModel	 Epoch: 13	 Loss: 0.2157	 Accuracy: 0.9262
# Model: StudentModel	 Epoch: 14	 Loss: 0.2105	 Accuracy: 0.9292
# Model: StudentModel	 Epoch: 15	 Loss: 0.2054	 Accuracy: 0.9305
# Model: StudentModel	 Epoch: 16	 Loss: 0.2015	 Accuracy: 0.9323
# Model: StudentModel	 Epoch: 17	 Loss: 0.1970	 Accuracy: 0.9311
# Model: StudentModel	 Epoch: 18	 Loss: 0.1940	 Accuracy: 0.9338
# Model: StudentModel	 Epoch: 19	 Loss: 0.1910	 Accuracy: 0.9313
# Model: StudentModel	 Epoch: 20	 Loss: 0.1879	 Accuracy: 0.9326
# Model: StudentModel	 Epoch: 21	 Loss: 0.1854	 Accuracy: 0.9352
# Model: StudentModel	 Epoch: 22	 Loss: 0.1821	 Accuracy: 0.9347
# Model: StudentModel	 Epoch: 23	 Loss: 0.1797	 Accuracy: 0.9365
# Model: StudentModel	 Epoch: 24	 Loss: 0.1776	 Accuracy: 0.9363
# Model: StudentModel	 Epoch: 25	 Loss: 0.1754	 Accuracy: 0.9360
# Model: StudentModel	 Epoch: 26	 Loss: 0.1730	 Accuracy: 0.9347
# Model: StudentModel	 Epoch: 27	 Loss: 0.1723	 Accuracy: 0.9388
# Model: StudentModel	 Epoch: 28	 Loss: 0.1699	 Accuracy: 0.9390
# Model: StudentModel	 Epoch: 29	 Loss: 0.1678	 Accuracy: 0.9370
# Model: StudentModel	 Epoch: 30	 Loss: 0.1664	 Accuracy: 0.9387
# Model: StudentModel	 Epoch: 31	 Loss: 0.1643	 Accuracy: 0.9411
# Model: StudentModel	 Epoch: 32	 Loss: 0.1624	 Accuracy: 0.9389
# Model: StudentModel	 Epoch: 33	 Loss: 0.1607	 Accuracy: 0.9323
# Model: StudentModel	 Epoch: 34	 Loss: 0.1586	 Accuracy: 0.9433
# Model: StudentModel	 Epoch: 35	 Loss: 0.1581	 Accuracy: 0.9328
# Model: StudentModel	 Epoch: 36	 Loss: 0.1567	 Accuracy: 0.9412
# Model: StudentModel	 Epoch: 37	 Loss: 0.1551	 Accuracy: 0.9427
# Model: StudentModel	 Epoch: 38	 Loss: 0.1538	 Accuracy: 0.9397
# Model: StudentModel	 Epoch: 39	 Loss: 0.1540	 Accuracy: 0.9427
# Model: StudentModel	 Epoch: 40	 Loss: 0.1514	 Accuracy: 0.9410
# Model: StudentModel	 Epoch: 41	 Loss: 0.1500	 Accuracy: 0.9419
# Model: StudentModel	 Epoch: 42	 Loss: 0.1494	 Accuracy: 0.9422
# Model: StudentModel	 Epoch: 43	 Loss: 0.1486	 Accuracy: 0.9406
# Model: StudentModel	 Epoch: 44	 Loss: 0.1479	 Accuracy: 0.9432
# Model: StudentModel	 Epoch: 45	 Loss: 0.1465	 Accuracy: 0.9449
# Model: StudentModel	 Epoch: 46	 Loss: 0.1458	 Accuracy: 0.9433
# Model: StudentModel	 Epoch: 47	 Loss: 0.1450	 Accuracy: 0.9454
# Model: StudentModel	 Epoch: 48	 Loss: 0.1436	 Accuracy: 0.9409
# Model: StudentModel	 Epoch: 49	 Loss: 0.1422	 Accuracy: 0.9417
# Model: StudentModel	 Epoch: 50	 Loss: 0.1417	 Accuracy: 0.9438
# Model 3 name: StudentModelB
# Model: StudentModel	 Epoch: 1	 Loss: 25.8413	 Accuracy: 0.8123
# Model: StudentModel	 Epoch: 2	 Loss: 21.2941	 Accuracy: 0.8629
# Model: StudentModel	 Epoch: 3	 Loss: 20.4467	 Accuracy: 0.8862
# Model: StudentModel	 Epoch: 4	 Loss: 20.0229	 Accuracy: 0.8997
# Model: StudentModel	 Epoch: 5	 Loss: 19.8189	 Accuracy: 0.9053
# Model: StudentModel	 Epoch: 6	 Loss: 19.6973	 Accuracy: 0.9088
# Model: StudentModel	 Epoch: 7	 Loss: 19.6146	 Accuracy: 0.9073
# Model: StudentModel	 Epoch: 8	 Loss: 19.5517	 Accuracy: 0.9099
# Model: StudentModel	 Epoch: 9	 Loss: 19.5018	 Accuracy: 0.9133
# Model: StudentModel	 Epoch: 10	 Loss: 19.4628	 Accuracy: 0.9127
# Model: StudentModel	 Epoch: 11	 Loss: 19.4307	 Accuracy: 0.9140
# Model: StudentModel	 Epoch: 12	 Loss: 19.3997	 Accuracy: 0.9160
# Model: StudentModel	 Epoch: 13	 Loss: 19.3690	 Accuracy: 0.9168
# Model: StudentModel	 Epoch: 14	 Loss: 19.3473	 Accuracy: 0.9177
# Model: StudentModel	 Epoch: 15	 Loss: 19.3250	 Accuracy: 0.9176
# Model: StudentModel	 Epoch: 16	 Loss: 19.3050	 Accuracy: 0.9187
# Model: StudentModel	 Epoch: 17	 Loss: 19.2838	 Accuracy: 0.9181
# Model: StudentModel	 Epoch: 18	 Loss: 19.2603	 Accuracy: 0.9213
# Model: StudentModel	 Epoch: 19	 Loss: 19.2405	 Accuracy: 0.9203
# Model: StudentModel	 Epoch: 20	 Loss: 19.2186	 Accuracy: 0.9231
# Model: StudentModel	 Epoch: 21	 Loss: 19.1941	 Accuracy: 0.9221
# Model: StudentModel	 Epoch: 22	 Loss: 19.1751	 Accuracy: 0.9239
# Model: StudentModel	 Epoch: 23	 Loss: 19.1520	 Accuracy: 0.9251
# Model: StudentModel	 Epoch: 24	 Loss: 19.1332	 Accuracy: 0.9254
# Model: StudentModel	 Epoch: 25	 Loss: 19.1127	 Accuracy: 0.9266
# Model: StudentModel	 Epoch: 26	 Loss: 19.0906	 Accuracy: 0.9269
# Model: StudentModel	 Epoch: 27	 Loss: 19.0702	 Accuracy: 0.9254
# Model: StudentModel	 Epoch: 28	 Loss: 19.0494	 Accuracy: 0.9269
# Model: StudentModel	 Epoch: 29	 Loss: 19.0310	 Accuracy: 0.9278
# Model: StudentModel	 Epoch: 30	 Loss: 19.0092	 Accuracy: 0.9298
# Model: StudentModel	 Epoch: 31	 Loss: 18.9906	 Accuracy: 0.9320
# Model: StudentModel	 Epoch: 32	 Loss: 18.9722	 Accuracy: 0.9313
# Model: StudentModel	 Epoch: 33	 Loss: 18.9575	 Accuracy: 0.9333
# Model: StudentModel	 Epoch: 34	 Loss: 18.9400	 Accuracy: 0.9337
# Model: StudentModel	 Epoch: 35	 Loss: 18.9270	 Accuracy: 0.9327
# Model: StudentModel	 Epoch: 36	 Loss: 18.9144	 Accuracy: 0.9355
# Model: StudentModel	 Epoch: 37	 Loss: 18.9011	 Accuracy: 0.9369
# Model: StudentModel	 Epoch: 38	 Loss: 18.8899	 Accuracy: 0.9357
# Model: StudentModel	 Epoch: 39	 Loss: 18.8747	 Accuracy: 0.9372
# Model: StudentModel	 Epoch: 40	 Loss: 18.8669	 Accuracy: 0.9371
# Model: StudentModel	 Epoch: 41	 Loss: 18.8547	 Accuracy: 0.9357
# Model: StudentModel	 Epoch: 42	 Loss: 18.8456	 Accuracy: 0.9376
# Model: StudentModel	 Epoch: 43	 Loss: 18.8330	 Accuracy: 0.9381
# Model: StudentModel	 Epoch: 44	 Loss: 18.8278	 Accuracy: 0.9372
# Model: StudentModel	 Epoch: 45	 Loss: 18.8168	 Accuracy: 0.9377
# Model: StudentModel	 Epoch: 46	 Loss: 18.8057	 Accuracy: 0.9377
# Model: StudentModel	 Epoch: 47	 Loss: 18.7981	 Accuracy: 0.9392
# Model: StudentModel	 Epoch: 48	 Loss: 18.7890	 Accuracy: 0.9379
# Model: StudentModel	 Epoch: 49	 Loss: 18.7821	 Accuracy: 0.9397
# Model: StudentModel	 Epoch: 50	 Loss: 18.7733	 Accuracy: 0.9383
#   0%|          | 0/1500 [00:00<?, ?it/s]Model 4 name: StudentModelC
# 100%|██████████| 1500/1500 [28:18<00:00,  1.13s/it]
# Model: Kmeans 	 Epoch: 1	 Loss: 0.0466	 Accuracy: 0.0416
# 100%|██████████| 1500/1500 [30:50<00:00,  1.23s/it]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 2	 Loss: 0.0229	 Accuracy: 0.0408
# 100%|██████████| 1500/1500 [13:32<00:00,  1.85it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 3	 Loss: 0.0193	 Accuracy: 0.0423
# 100%|██████████| 1500/1500 [13:11<00:00,  1.89it/s]
# Model: Kmeans 	 Epoch: 4	 Loss: 0.0179	 Accuracy: 0.0408
# 100%|██████████| 1500/1500 [12:53<00:00,  1.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 5	 Loss: 0.0171	 Accuracy: 0.0475
# 100%|██████████| 1500/1500 [12:59<00:00,  1.92it/s]
# Model: Kmeans 	 Epoch: 6	 Loss: 0.0166	 Accuracy: 0.0456
# 100%|██████████| 1500/1500 [13:09<00:00,  1.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 7	 Loss: 0.0162	 Accuracy: 0.0421
# 100%|██████████| 1500/1500 [13:19<00:00,  1.88it/s]
# Model: Kmeans 	 Epoch: 8	 Loss: 0.0159	 Accuracy: 0.0436
# 100%|██████████| 1500/1500 [12:51<00:00,  1.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 9	 Loss: 0.0157	 Accuracy: 0.0412
# 100%|██████████| 1500/1500 [12:41<00:00,  1.97it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 10	 Loss: 0.0154	 Accuracy: 0.0393
# 100%|██████████| 1500/1500 [12:54<00:00,  1.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 11	 Loss: 0.0152	 Accuracy: 0.0359
# 100%|██████████| 1500/1500 [13:09<00:00,  1.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 12	 Loss: 0.0151	 Accuracy: 0.0385
# 100%|██████████| 1500/1500 [13:00<00:00,  1.92it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 13	 Loss: 0.0150	 Accuracy: 0.0416
# 100%|██████████| 1500/1500 [14:42<00:00,  1.70it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 14	 Loss: 0.0148	 Accuracy: 0.0398
# 100%|██████████| 1500/1500 [32:51<00:00,  1.31s/it]
# Model: Kmeans 	 Epoch: 15	 Loss: 0.0147	 Accuracy: 0.0388
# 100%|██████████| 1500/1500 [30:33<00:00,  1.22s/it]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 16	 Loss: 0.0146	 Accuracy: 0.0378
# 100%|██████████| 1500/1500 [30:45<00:00,  1.23s/it]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 17	 Loss: 0.0145	 Accuracy: 0.0333
# 100%|██████████| 1500/1500 [30:57<00:00,  1.24s/it]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 18	 Loss: 0.0144	 Accuracy: 0.0337
# 100%|██████████| 1500/1500 [31:39<00:00,  1.27s/it]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 19	 Loss: 0.0143	 Accuracy: 0.0356
# 100%|██████████| 1500/1500 [31:03<00:00,  1.24s/it]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 20	 Loss: 0.0142	 Accuracy: 0.0362
# 100%|██████████| 1500/1500 [24:05<00:00,  1.04it/s]
# Model: Kmeans 	 Epoch: 21	 Loss: 0.0142	 Accuracy: 0.0410
# 100%|██████████| 1500/1500 [22:19<00:00,  1.12it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 22	 Loss: 0.0141	 Accuracy: 0.0345
# 100%|██████████| 1500/1500 [33:51<00:00,  1.35s/it]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 23	 Loss: 0.0139	 Accuracy: 0.0418
# 100%|██████████| 1500/1500 [31:45<00:00,  1.27s/it]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 24	 Loss: 0.0138	 Accuracy: 0.0346
# 100%|██████████| 1500/1500 [09:07<00:00,  2.74it/s]
# Model: Kmeans 	 Epoch: 25	 Loss: 0.0138	 Accuracy: 0.0377
# 100%|██████████| 1500/1500 [08:36<00:00,  2.90it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 26	 Loss: 0.0137	 Accuracy: 0.0376
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 27	 Loss: 0.0136	 Accuracy: 0.0369
# 100%|██████████| 1500/1500 [08:41<00:00,  2.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 28	 Loss: 0.0135	 Accuracy: 0.0382
# 100%|██████████| 1500/1500 [09:45<00:00,  2.56it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 29	 Loss: 0.0135	 Accuracy: 0.0383
# 100%|██████████| 1500/1500 [09:03<00:00,  2.76it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 30	 Loss: 0.0134	 Accuracy: 0.0367
# 100%|██████████| 1500/1500 [08:40<00:00,  2.88it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 31	 Loss: 0.0134	 Accuracy: 0.0372
# 100%|██████████| 1500/1500 [08:29<00:00,  2.94it/s]
# Model: Kmeans 	 Epoch: 32	 Loss: 0.0133	 Accuracy: 0.0387
# 100%|██████████| 1500/1500 [08:28<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 33	 Loss: 0.0133	 Accuracy: 0.0357
# 100%|██████████| 1500/1500 [08:29<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 34	 Loss: 0.0132	 Accuracy: 0.0387
# 100%|██████████| 1500/1500 [08:29<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 35	 Loss: 0.0132	 Accuracy: 0.0391
# 100%|██████████| 1500/1500 [08:29<00:00,  2.94it/s]
# Model: Kmeans 	 Epoch: 36	 Loss: 0.0131	 Accuracy: 0.0395
# 100%|██████████| 1500/1500 [08:32<00:00,  2.93it/s]
# Model: Kmeans 	 Epoch: 37	 Loss: 0.0130	 Accuracy: 0.0373
# 100%|██████████| 1500/1500 [08:26<00:00,  2.96it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 38	 Loss: 0.0130	 Accuracy: 0.0371
# 100%|██████████| 1500/1500 [08:30<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 39	 Loss: 0.0129	 Accuracy: 0.0387
# 100%|██████████| 1500/1500 [08:29<00:00,  2.94it/s]
# Model: Kmeans 	 Epoch: 40	 Loss: 0.0129	 Accuracy: 0.0393
# 100%|██████████| 1500/1500 [08:29<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 41	 Loss: 0.0128	 Accuracy: 0.0375
# 100%|██████████| 1500/1500 [08:28<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 42	 Loss: 0.0127	 Accuracy: 0.0391
# 100%|██████████| 1500/1500 [08:28<00:00,  2.95it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 43	 Loss: 0.0127	 Accuracy: 0.0396
# 100%|██████████| 1500/1500 [08:30<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 44	 Loss: 0.0126	 Accuracy: 0.0367
# 100%|██████████| 1500/1500 [08:31<00:00,  2.93it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 45	 Loss: 0.0126	 Accuracy: 0.0385
# 100%|██████████| 1500/1500 [08:30<00:00,  2.94it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 46	 Loss: 0.0126	 Accuracy: 0.0435
# 100%|██████████| 1500/1500 [08:42<00:00,  2.87it/s]
# Model: Kmeans 	 Epoch: 47	 Loss: 0.0125	 Accuracy: 0.0376
# 100%|██████████| 1500/1500 [08:46<00:00,  2.85it/s]
#   0%|          | 0/1500 [00:00<?, ?it/s]Model: Kmeans 	 Epoch: 48	 Loss: 0.0125	 Accuracy: 0.0413
# 100%|██████████| 1500/1500 [08:42<00:00,  2.87it/s]
# Model: Kmeans 	 Epoch: 49	 Loss: 0.0125	 Accuracy: 0.0399
# 100%|██████████| 1500/1500 [08:40<00:00,  2.88it/s]
# Model: Kmeans 	 Epoch: 50	 Loss: 0.0124	 Accuracy: 0.0381
# Save plot as ./alpha_value/alpha_value_0.0.png
#
# Process finished with exit code 0

