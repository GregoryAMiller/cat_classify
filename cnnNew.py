import numpy as np
import pandas as pd
from PIL import Image
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as AF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        # Store the data and transformation
        self.data = df
        self.transform = transform
        
    def __len__(self):
        # Return the length of the dataset
        return len(self.data)
    
    def __getitem__(self, index):
        # Get the data sample
        sample = self.data.iloc[index]
        
        # Apply the transformation, if any
        if self.transform:
            sample = self.transform(sample)
        
        return sample



# Create an empty DataFrame to store the data
df = pd.DataFrame()

# Loop through all the image files in the folder
for file in os.listdir("C:/Users/Grego/Desktop/cat_classify/images"):
    
  # Load the image file
  image = Image.open(f'C:/Users/Grego/Desktop/cat_classify/images/{file}')
  print(f'Loading {file}.......')
  
  # Preprocess the image as needed (e.g. resize, convert to grayscale, normalize)
  # image = np.array(image) / 255
  
  # Flatten the image into a 1D array of pixels
  # pixels = image.flatten()
  
  # Extract the label from the file name (assumes the file name is in the format "image_<label>.jpg")
  label = int(file.split("_")[1].split(".")[0])
  
  # Create a new row with one column for each pixel
  # row = {}
  # for i, pixel in enumerate(pixels):
  #   row["pixel_{}".format(i)] = pixel
  row["image"] = image
  row["label"] = label
  
  # Concatenate the new row to the DataFrame
  df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
# Read the CSV file
# df = pd.read_csv('C:/Users/Grego/Documents/chat_GPT/cat_classify/data.csv', sep=',', header=0, names=None, index_col=None, usecols=None)
print(df.head)
# Define the transformation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Create the dataset
dataset = DataFrameDataset(df, transform=transform)

# Use the dataset
for i in range(len(dataset)):
    sample = dataset[i]
    print(sample)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Define the pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Define the activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Apply the convolutional layers and pooling
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten the feature maps
        x = x.view(-1, 64 * 7 * 7)
        
        # Apply the fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

# Create the CNN model
model = CNN(num_classes=3)

# Define the loss function, optimizer, and metric
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
metric = nn.Accuracy()

# Define the number of epochs
num_epochs = 10

# Define the training and testing loop
for epoch in range(num_epochs):
    # Training loop
    for images, labels in mnist_train:
        # Clear the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute the loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update the weights and biases
        optimizer.step()
        
        # Compute the metric
        metric.update((outputs, labels))
        
    # Print the metric and reset it for the next epoch
    print(f"Epoch {epoch+1} - Training: {metric.compute():.4f}")
    metric.reset()
    
    # Testing loop
    for images, labels in mnist_test:
        # Forward pass
        outputs = model(images)
        
        # Compute the loss
        loss = criterion(outputs, labels)
        
        # Compute the metric
        metric.update((outputs, labels))
        
    # Print the metric
    print(f"Epoch {epoch+1} - Testing: {metric.compute():.4f}")
    metric.reset()