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
p=argparse.ArgumentParser(description="Parses model predictions configurations")

# Settings for command line arguments
p.add_argument('--checkpoint_dir', action='store', default='save_dir/checkpoint1.pth', type=str, help='Provide Path to Checkpoint File')
p.add_argument('--image_path', action='store', type=str,default='flowers/train/10/image_07098.jpg', help='Provide Path to Image')
p.add_argument('--top_k', action='store', default=5, type=int, help='Provide Number of top K Values to Fetch')
p.add_argument('--json_file', action='store', default='cat_to_name.json', type=str, help='Provide Category Name to Label Mapping File')
p.add_argument('--gpu', action='store', default='gpu', type=str, help='Choose between CPU and GPU')
args=p.parse_args()

# Create variable names consistent with the code
checkpoint_file=args.checkpoint_dir
image_file=args.image_path
top_k=args.top_k
json_file=args.json_file
dev=args.gpu

# Print summary of model configurations used
print('Parsed Parameters Summary: ',
      '\n Checkpoint_File: ',checkpoint_file,
      '\n Image_Path: ', image_file,
      '\n Top_k: ',top_k,
      '\n Json_File: ',json_file,
      '\n GPU: ',dev    
     )

#Choose between GPU and CPU
if dev == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'


with open(json_file, 'r') as f:
    cat_to_name = json.load(f)
    
# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    cp=torch.load(filepath)
    model=cp['model']
    epochs=cp['epochs']
    print(epochs)
    batch_size=cp['batch_size']
    model.classifier=cp['classifier']
    optimizer=cp['optimizer']
    model.load_state_dict(cp['state_dict'])
    model.class_to_idx=cp['class_to_idx']
    
    return model, cp['class_to_idx'],cp['optimizer'], cp['class_to_idx']
   
# Call the function to build the model
model, class_to_idx, optimizer, class_index = load_checkpoint(checkpoint_file)

# Function to Preprocess Image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    img_preprocess = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    image=img_preprocess(image)
    return image

# Function to Predict Class and Prbabilities
def predict(image_path, model, topk=top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    model.eval()
    
    # Read and process image
    pic=Image.open(image_path)
    pic=process_image(pic)
    
    # Convert to vector and then to torch tensor
    pic=np.expand_dims(pic, 0)
    pic=torch.from_numpy(pic)
    
    # Define Inputs
    inputs=pic.to(device)
    
    # Feedforward part
    logps=model.forward(inputs)          # Calculates log probabilities
    ps=F.softmax(logps,dim=1)            # Use Softmax to get probabilities
    

    # Check Predictions and get top 5 probabilities
    top_p, top_class = ps.cpu().topk(topk,dim=1)                # Find highest probabilities and its class
    
    # Get top Probabilities and their corresponding Classes
    top_p = top_p.data.numpy().squeeze().tolist()
    top_class= top_class.data.numpy().squeeze().tolist()
    
    return top_p, top_class


# Select image
impth=image_file

# Test and run the predict function on selected image file
probs, classes = predict(impth, model.to(device))
# classes = [class_index[x] for x in classes]

# Match with correct index using class_to_idx
get_keys = []
for i in classes:
    for key, value in class_index.items():
        if value == i:
            get_keys.append(key)

# Get Flower Names of top classes from cat_to_names using get_keys
flower_names=[cat_to_name[x] for x in get_keys]

# Print outcomes
print('Probabilities:', probs)
print('Classes:',classes)
print('Keys:',get_keys)
print('Flower Name:',flower_names)