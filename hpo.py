#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import os
import logging
import sys
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion):
    model.eval()        # for testing using evalualion function
    running_loss=0      # assign running loss
    running_corrects=0  # assign running corrects
    
    for inputs, labels in test_loader:
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)            # calculate running loss
        running_corrects += torch.sum(preds == labels.data)     # calculate running corrects

    total_loss = running_loss // len(test_loader)       
    total_acc = running_corrects.double() // len(test_loader)
    
    logger.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}\n".format(
            total_loss, total_acc)
        ) # print the loss and accuracy values
    

def train(model, trainloader, testloader, loss_criterion, optimizer):
    running_loss = 0
    correct_pred = 0
    epochs = 4

    for e in range(epochs):
    
        model.train()
        
        for inputs, labels in trainloader:
            optimizer.zero_grad() # freeze the model's optimizer
            output = model(inputs)
            loss = loss_criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            # Training Loss
            _, preds = torch.max(output, 1)
            running_loss += loss.item() * inputs.size(0)
            correct_pred += torch.sum(preds == labels.data)

        epoch_loss = running_loss // len(trainloader)
        epoch_acc = correct_pred // len(trainloader)
        
        logger.info("\nEpoch: {}/{}.. ".format(e+1, epochs))
        logger.info("Training Loss: {:.4f}".format(epoch_loss))
        logger.info("Training Accuracy: {:.4f}".format(epoch_acc))
        # print the Training loss and Accuracy as each epoch
            
        test(model, testloader, loss_criterion)    
    
    return model 
    
def net():

    model = models.resnet34(pretrained=True) # using the pretrained resnet34 model with 34 layers

    for param in model.parameters():
        param.requires_grad = False # freeze the model

    model.fc = nn.Sequential(
                   nn.Linear(512, 256), # adding own NN layers to the output of the pretrained model
                   nn.ReLU(inplace=True),
                   nn.Linear(256, 133)) # output should be 133 as we have 133 classes of dog breeds

     
    return model

def create_data_loaders(data, batch_size):
    
    train_data_path = os.path.join(data, 'train') # Calling OS Environment variable and split it into 3 sets
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')
    
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]) # transforming the training image data
                                                            
    test_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]) # transforming the testing image data

    # loading train,test & validation data from S3 location using torchvision datasets' Imagefolder function
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size) 
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args): # args to use with jypyter notebook's Estimater function
    
        
    train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batchsize)
   
    model=net() # Initialize a model by calling the net function

    loss_criterion = nn.CrossEntropyLoss() # using cross Entropy loss function
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr) #using adam optimizer
    
    logger.info("Start Model Training")
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer) # calling the train function to start the training
    
    logger.info("Testing Model")
    test(model, test_loader, loss_criterion) # testing model
    

    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth")) # save the trained model to S3

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser() # adding the args parsers to use with the notebook estimator call

    
    parser.add_argument(
        "--batchsize",
        type = int,
        default = 64,
        metavar = "N",
        help = "input batch size for training (default: 64)",
    )
    
    parser.add_argument(
        "--lr", type = float, default = 0.1, metavar = "LR", help = "learning rate (default: 1.0)"
    )
    # Using sagemaker OS Environ's channels to locate the training data, model dir and output dir to save in S3 bucket.
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAIN']) 
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args = parser.parse_args()

    main(args)
