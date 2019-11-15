# Import libraries
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
from workspace_utils import active_session
import PIL
from PIL import Image
import numpy as np
import argparse
import json

# Arguments parsing section
p=argparse.ArgumentParser(description="Parses model training configurations")

# Settings for command line arguments
p.add_argument('--data_dir', action='store', default='flowers', type=str, help='Provide Path to Flowers Directory')
p.add_argument('--arch', action='store', default='densenet161', type=str, help='Choose between vgg13 or densenet161')
p.add_argument('--hidden_units', action='store', default=1000, type=int, help='Specify Number of Hidden Units')
p.add_argument('--learning_rate', action='store', default=0.003, type=float, help='Specify Learning Rate')
p.add_argument('--epochs', action='store', default=2, type=int, help='Specify Epochs')
p.add_argument('--gpu', action='store', default='gpu', type=str, help='Choose between CPU and GPU')
p.add_argument('--save_dir', action='store', default='save_dir', type=str, help='Provide Path to Save Checkpoint')
args=p.parse_args()

# Create variable names consistent with the code
dd=args.data_dir
arch=args.arch
hu=args.hidden_units
alpha=args.learning_rate
ep=args.epochs
dev=args.gpu
save_dir=args.save_dir

# Print summary of model configurations used
print('Parsed Parameters Summary: ',
      '\n Data_Directory: ',dd,
      '\n Model: ', arch,
      '\n Hidden Units: ',hu,
      '\n Learning Rate: ',alpha,
      '\n Epochs: ',ep,
      '\n Device: ',dev,
      '\n Save: ',save_dir      
     )
   

#Loading Data
try:
    # Define data directories
    data_dir=dd
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)
    test_data  = datasets.ImageFolder(test_dir,transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
except Exception as e:
    print(e)

# Map labels of flowers to their respective name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


#Choose between GPU and CPU
if dev == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'


# Load Models either densenet161 or vgg13
if arch == 'densenet161':
    model=models.densenet161(pretrained=True)
    
    #Freeze model parameters to avoid backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    #Create a feedforward classifier, Loss Fuction and Optimizer
    # Define model inputs
    input_layer = 2208;    # From model architecture of vgg16 model
    out_layer=102;         # Number of flower categories to predict
    h1=hu;                 # Features in hidden layer 1
    
else:
    model=models.vgg13(pretrained=True)
    
        #Freeze model parameters to avoid backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    #Create a feedforward classifier, Loss Fuction and Optimizer
    # Define model inputs
    input_layer = 25088;    # From model architecture of vgg16 model
    out_layer=102;          # Number of flower categories to predict
    h1=hu;                  # Features in hidden layer 1

# Print Model Details
# print("Our model: \n\n", model, '\n')

# Define Classifier
model.classifier = nn.Sequential(OrderedDict([
                                            ('fc1',nn.Linear(input_layer,h1)),
                                            ('relu1',nn.ReLU()),
                                            ('drop1',nn.Dropout(0.2)),

                                            ('fc2',nn.Linear(h1,out_layer)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))
# Define Loss Function
criterion=nn.NLLLoss()
# Define Optimizer
optimizer = optim.Adam(model.classifier.parameters(), lr=alpha)
# Assign to GPU/CPU
model.to(device);


# Train the Network
with active_session():
    epochs=ep      # Number of iteration per epoch = number_of_samples / batch_size
    steps=0
    running_loss=0
    print_every=10

    for i in range(epochs):
        # Training Loop
        for inputs, labels in train_loader:
            steps+=1
            inputs,labels =inputs.to(device),labels.to(device)

            # Clear gradients while in training loop
            optimizer.zero_grad()

            # Feedforward part
            logps=model.forward(inputs)    # Calculates log probabilities
            loss=criterion(logps,labels)   # Calculates loss using error function (between prediction and actual)

            # Backward propagation part
            loss.backward()     # Project errors backwards (Note: Model parameters are frozen for pretrained models)
            optimizer.step()    # Optimize using gradient descent

            # Calculate Training loss
            running_loss += loss.item()

            # Logic to enter Validation Loop
            if steps % print_every ==0:
                valid_loss=0
                accuracy=0
                model.eval()    # Set network to evaluation mode

                #Validation Loop
                with torch.no_grad():
                    for inputs,labels in valid_loader:
                        inputs,labels =inputs.to(device), labels.to(device)

                        # Feedforward part
                        logps=model.forward(inputs)        # Calculates log probabilities
                        batch_loss=criterion(logps,labels) # Calculates Batch Loss

                        # Calculate Validation loss
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps=torch.exp(logps)                                # Get probabilities from log probabilities
                        top_p, top_class = ps.topk(1,dim=1)                # Find highest probability and its class
                        equals=top_class == labels.view(*top_class.shape)  # 1 where correct 0 where wrong

                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item() # Take mean of equals

                # Print Progress
                print(f"Epoch {i+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

                #Set Running loss to 0 and switch model back to training mode
                running_loss = 0
                model.train()
                
# Testing the network
test_loss=0
accuracy=0
model.eval()    # Set network to evaluation mode

#Test Loop
with torch.no_grad():
    for inputs,labels in test_loader:
        inputs,labels =inputs.to(device), labels.to(device)

        # Feedforward part
        logps=model.forward(inputs)        # Calculates log probabilities
        batch_loss=criterion(logps,labels) # Calculates Batch Loss

        # Calculate Validation loss
        test_loss += batch_loss.item()

        # Calculate accuracy
        ps=torch.exp(logps)                                # Get probabilities from log probabilities
        top_p, top_class = ps.topk(1,dim=1)                # Find highest probability and its class
        equals=top_class == labels.view(*top_class.shape)  # 1 where correct 0 where wrong

        accuracy += torch.mean(equals.type(torch.FloatTensor)).item() # Calculated by taking mean of equals

# Print Test Accuracy
print(f"Test loss: {test_loss/len(test_loader):.3f}.. "
      f"Test accuracy: {accuracy/len(test_loader):.3f}")

# Save the checkpoint
model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': input_layer,
              'output_size': out_layer,
              'hidden_layers': h1,
              'model':model,
              'epochs':epochs,
              'batch_size': 64,
              'classifier': model.classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx
             }

# Save Checkpoint
try:
    torch.save(checkpoint, save_dir+'/checkpoint.pth')
except:
    torch.save(checkpoint,'checkpoint.pth')