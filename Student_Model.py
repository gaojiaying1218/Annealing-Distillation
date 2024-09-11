import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformation for the images (e.g., resize, normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Specify the path to the root directory of your dataset
dataset_path = 'D:\DataSet\JiDaTop5/'

# Create an ImageFolder dataset
dataset = ImageFolder(root=dataset_path, transform=transform)

# Split the dataset into training and test datasets
train_size = int(0.8 * len(dataset))  # 80% of data for training, 20% for testing
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


# Create a DataLoader to load the dataset
# You can adjust batch_size and other parameters as needed


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



#######

class StudentModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super(StudentModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        # = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        # =self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x

model = StudentModel()
model = model.to(device)

summary(model)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

epochs = 3
for epoch in range(epochs):
    model.train()
    # 训练集上训练模型权重
    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        # 前向预测
        preds = model(data)
        loss = criterion(preds, targets)
        # 反向传播，优化权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 测试集上评估模型性能
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct / num_samples).item()
    model.train()
    print('Epoch:{}\t Accuracy:{:.4f}'.format(epoch + 1, acc))

student_model_scratch = model
