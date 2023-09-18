import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision import datasets, transforms
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as AF
from torch.utils.data import DataLoader
import time

#train and test data directory
data_dir = "C:/Users/Grego/Desktop/cat_classify/images"

# To display some images
def show_some_digit_images(images):
    print("> Shapes of image:", images.shape)
    #print("Matrix for one image:")
    #print(images[1][0])
    for i in range(0, 10):
        plt.subplot(2, 5, i+1) # Display each image at i+1 location in 2 rows and 5 columns (total 2*5=10 images)
        plt.imshow(images[i][0]) # show ith image from image matrices by color map='Oranges'
    plt.show()

############################### ANN modeling #################################
### Convert the image into numbers: transforms.ToTensor()
# It separate the image into three color channels RGB and converts the pixels of each images to the brightness
# of the color in the range [0,255] that are scaled down to a range [0,1]. The image is now a Torch Tensor (array object)
### Normalize the tensor: transforms.Normalize() normalizes the tensor with mean (0.5) and stdev (0.5)
#+ You can change the mean and stdev values
print("------------------ANN modeling---------------------------")

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# PyTorch tensors are like NumPy arrays that can run on GPU
# e.g., x = torch.randn(64,100).type(dtype) # need to cast tensor to a CUDA datatype (dtype)

from torch.autograd import Variable
x = Variable

#load the train and test data
train_dataset = ImageFolder(data_dir, transform = transforms)
test_dataset = ImageFolder(data_dir, transform = transforms)

# You can use random_split function to splite a dataset
#from torch.utils.data.dataset import random_split
#train_data, val_data, test_data = random_split(train_dataset, [60,20,20])

### DataLoader will shuffle the training dataset and load the training and test dataset
mini_batch_size = 40 #+ You can change this mini_batch_size
# If mini_batch_size==100, # of training batches=6000/100=600 batches, each batch contains 100 samples (images, labels)
# DataLoader will load the data set, shuffle it, and partition it into a set of samples specified by mini_batch_size.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=mini_batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=mini_batch_size, shuffle=True)


### Let's display some images from the first batch to see what actual digit images look like
iterable_batches = iter(train_dataloader) # making a dataset iterable
images, labels = next(iterable_batches) # If you can call next() again, you get the next batch until no more batch left
show_digit_image = True
if show_digit_image:
    show_some_digit_images(images)

### CNN architecture
class CNN(nn.Module):
    def __init__(self, dropout_pr, num_hidden, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*25*25, 512)
        self.fc2 = nn.Linear(512, 3)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Training function
def train_ANN_model(num_epochs, training_data, device, CUDA_enabled, is_MLP, ANN_model, loss_func, optimizer):
    if (device.type == 'cuda' and CUDA_enabled):
        print("...Modeling using GPU...")        
        ANN_model = ANN_model.to(device=device) # sending to whaever device (for GPU acceleration)
        print("Memory allocated:", torch.cuda.memory_allocated(cuda_id))
        print("Memory reserved:", torch.cuda.memory_reserved(cuda_id))
        if (next(ANN_model.parameters()).is_cuda):
            print("ANN_model is on GPU...")
    else:
        print("...Modeling using CPU...")

    train_losses = []
    for epoch_cnt in range(num_epochs):
        ANN_model.train() # to set the model in training mode. Only Dropout and BatchNorm care about this flag.
        for batch_cnt, (images, labels) in enumerate(training_data):
            # Each batch contain batch_size (100) images, each of which 1 channel 28x28
            # print(images.shape) # the shape of images=[100,1,28,28]
            # So, we need to flatten the images into 28*28=784
            # -1 tells NumPy to flatten to 1D (784 pixels as input) for batch_size images
            if (is_MLP):
                # the size -1 is inferred from other dimensions
                images = images.reshape(-1, 784) # or images.view(-1, 784) or torch.flatten(images, start_dim=1)

            if (device.type == 'cuda' and CUDA_enabled):
                images = images.to(device) # moving tensors to device
                labels = labels.to(device)

            optimizer.zero_grad() # set the cumulated gradient to zero
            output = ANN_model(images) # feedforward images as input to the network
            loss = loss_func(output, labels) # computing loss
            #print("Loss: ", loss)
            #print("Loss item: ", loss.item())
            train_losses.append(loss.item())
            # PyTorch's Autograd engine (automatic differential (chain rule) package) 
            loss.backward() # calculating gradients backward using Autograd
            optimizer.step() # updating all parameters after every iteration through backpropagation
            # Display the training status
            if (batch_cnt+1) % mini_batch_size == 0:
                print(f"Epoch={epoch_cnt+1}/{num_epochs}, batch={batch_cnt+1}/{num_train_batches}, loss={loss.item()}")
    return train_losses

# Testing function
def test_ANN_model(device, CUDA_enabled, is_MLP, ANN_model, testing_data):
    # torch.no_grad() is a decorator for the step method
    # making "require_grad" false since no need to keeping track of gradients    
    predicted_digits=[]
    # torch.no_grad() deactivates Autogra engine (for weight updates). This help run faster
    with torch.no_grad():
        ANN_model.eval() # # set the model in testing mode. Only Dropout and BatchNorm care about this flag.
        for batch_cnt, (images, labels) in enumerate(testing_data):
            if (is_MLP):
                images = images.reshape(-1, 784) # or images.view(-1, 784) or torch.flatten(images, start_dim=1)

            if (device.type == 'cuda' and CUDA_enabled):
                images = images.to(device) # moving tensors to device
                labels = labels.to(device)
            
            output = ANN_model(images)
            _, prediction = torch.max(output,1) # returns the max value of all elements in the input tensor
            predicted_digits.append(prediction)
            num_samples = labels.shape[0]
            num_correct = (prediction==labels).sum().item()
            accuracy = num_correct/num_samples
            if (batch_cnt+1) % mini_batch_size == 0:
                print(f"batch={batch_cnt+1}/{num_test_batches}")
        print("> Number of samples=", num_samples, "number of correct prediction=", num_correct, "accuracy=", accuracy)
    return predicted_digits

### Create an object for the ANN model defined in the MLP class
# Architectural parameters: You can change these parameters except for num_input and num_classes
num_input = 200*200   # 28X28=784 pixels of image
num_classes = 3    # output layer
num_hidden = 1000     # number of neurons at the first hidden layer
# Randomly selected neurons by dropout_pr probability will be dropped (zeroed out) for regularization.
dropout_pr = 0.05

# CNN model
CNN_model = CNN(dropout_pr, num_hidden, num_classes)
print("> CNN model parameters")
print(CNN_model.parameters)

### Define a loss function: You can choose other loss functions
# loss_func = nn.L1Loss()
# loss_func = nn.MSELoss()
# loss_func = nn.GaussianNLLLoss()
# loss_func = nn.MSELoss()

loss_func = nn.CrossEntropyLoss()

### Choose a gradient method
# model hyperparameters and gradient methods
# optim.SGD performs gradient descent and update the weigths through backpropagation.
num_epochs = 1
alpha = 0.5    # learning rate (MLP: 0.1) (CNN: 0.3)
gamma = 0.1        # momentum (MLP: 0.5) (CNN: 0.1)

# CNN optimizer
# Different gradient methods used for CNN
# CNN_optimizer = optim.RMSprop(CNN_model.parameters(), lr=alpha, momentum=gamma) 
# CNN_optimizer = optim.Adadelta(CNN_model.parameters(), lr=alpha) 
# CNN_optimizer = optim.Adagrad(CNN_model.parameters(), lr=alpha) 
# CNN_optimizer = optim.ASGD(CNN_model.parameters(), lr=alpha) 
# CNN_optimizer = optim.Adam(CNN_model.parameters(), lr=alpha) 
CNN_optimizer = optim.SGD(CNN_model.parameters(), lr=alpha, momentum=gamma) 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CUDA_enabled = False
### Train your networks
print("............Training CNN................")
is_MLP = False
cnn_time_start = time.perf_counter()
train_loss=train_ANN_model(num_epochs, train_dataloader, device, CUDA_enabled, is_MLP, CNN_model, loss_func, CNN_optimizer)
cnn_time_stop = time.perf_counter()
print(f'cnn training time: {cnn_time_stop - cnn_time_start}')
print("............Testing CNN model................")
predicted_digits=test_ANN_model(device, CUDA_enabled, is_MLP, CNN_model, test_dataloader)
print("> Predicted digits by CNN model")
# print(predicted_digits) # comment out for testing

#To save and load a model
print("...Saving model...")
torch.save(CNN_model, 'CNN_model_CATS.pt')
pretrained_model = torch.load('CNN_model_CATS.pt')

import os
import random
import imageio

# Set the root folder
root_folder = data_dir

# Initialize an empty list to store the images
test_images = []

# Loop through the directories in the root folder
for directory in os.listdir(root_folder):
    # Construct the full path to the directory
    dir_path = os.path.join(root_folder, directory)
    # Check if the path is a directory
    if os.path.isdir(dir_path):
        # List the files in the directory
        files = os.listdir(dir_path)
        # Select 10 random files
        sample = random.sample(files, 10)
        # Read the images and append them to the list
        for file in sample:
            file_path = os.path.join(dir_path, file)
            image = imageio.imread(file_path)
            test_images.append(image)

# The images list should now contain 30 images (10 from each folder)
# print(len(test_images))
test_dir = 'C:/Users/Grego/Desktop/cat_classify/test'
train_dataset2 = ImageFolder(test_dir, transform = transforms)
train_dataloader2 = DataLoader(dataset=train_dataset2, shuffle=True)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
class_labels = ('bree', 'asha' 'annie')

dataiter = iter(train_dataloader2)
images, labels = dataiter.next()
print(labels)

# for image in images:
#     imshow(torchvision.utils.make_grid(image))
# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{class_labels[labels[j]]:5s}' for j in range(9)))
outputs = CNN_model(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%s' % class_labels[predicted[j]] for j in range(9)))
# print(labels)

# Set the model to evaluation mode
CNN_model.eval()
print(pretrained_model)
# Predict the class of each image
predictions = []
for image in test_images:
    image = torchvision.transforms.ToTensor()(image)
    # Expand the dimensions of the tensor to match the model's expected input shape
    image = image.unsqueeze(0)  # add a batch dimension
    output = CNN_model(image)
    # print(output)
    class_idx = torch.argmax(output)
    predictions.append(class_idx)
    #print(predictions)
# Plot the images and their predicted classes
fig = plt.figure(figsize=(10, 3))
for i, (image, prediction) in enumerate(zip(test_images, predictions)):
    ax = fig.add_subplot(10, 3, i+1)
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(f'Predicted Class: {class_labels[prediction]}')
plt.show()