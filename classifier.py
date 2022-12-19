import torch
import torch.nn as nn
import torch, torchvision
from torchvision import models
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt_1
import time
import os

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = models.resnet50()
    def train_model(self, num_classes):
        self.model = models.resnet50(pretrained=True)
        return self.func(num_classes)
    
    def retrain_model(self, num_classes):
        self.model = models.resnet50()
        return self.func(num_classes)
    
    def func(self, num_classes):
        for param in self.model.parameters():
            param.requires_grad = False
        fc_inputs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes), # Since 10 possible outputs
            nn.LogSoftmax(dim=1) # For using NLLLoss()
        )
        return self.model
    

def preprocessing(dataset_folder, batch_size):

    image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=[640,480], scale=(0.8, 1.0)),
        # transforms.RandomRotation(degrees=15),
        # transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(size=448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=[640,480]),
        # transforms.CenterCrop(size=448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=[640,480]),
        # transforms.CenterCrop(size=448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    }

    train_directory = os.path.join(dataset_folder, 'train')
    valid_directory = os.path.join(dataset_folder, 'valid')

    # Number of classes
    num_classes = len(os.listdir(valid_directory))  #10#2#257
    print("num_classes: ", num_classes)

    # Load Data from folders
    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
        # 'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
    }

    # Get a mapping of the indices to the class names, in order to see the output classes of the test images.
    idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
    print(idx_to_class)

    # Size of Data, to be used for calculating Average Loss and Accuracy
    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])

    # Create iterators for the Data loaded using DataLoader module
    train_data_loader = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)

    print("train_data_size: ", train_data_size)
    print("valid_data_size: ", valid_data_size)

    paras = [train_data_size ,valid_data_size,train_data_loader, valid_data_loader, num_classes]
    return paras

def train(epochs, paras, restored_path = '', output_path = ''):

    train_data_size ,valid_data_size,train_data_loader, valid_data_loader,num_classes = paras
    graph = True
    
    net= Classifier()
    if restored_path:
      model = net.retrain_model(num_classes)
    else:
      print("Transfer learning...")
      model = net.train_model(num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters())
    # Define Optimizer and Loss Function
    loss_criterion = nn.NLLLoss()
    model = model.to(device)
    start_epoch = 0
    if restored_path:
        print("ReTrianing...")
        checkpoint = torch.load(restored_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = int(checkpoint['epoch'])
        loss = checkpoint['loss']
        epochs += start_epoch

    '''
    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''

    start = time.time()
    history = []
    best_loss = 100000.0
    best_epoch = None

    for epoch in range(start_epoch,epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_data_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)
            # print("inputs: ", inputs.shape)
            # print("labels: ", labels)
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_criterion(outputs, labels)
            
            # Backpropagate the gradients
            # loss.requires_grad = True
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
            #print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        
        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch + 1
            torch.save(model,f"{output_path}/exported_best_model.pt")

        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_data_size 
        avg_valid_acc = valid_acc/valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if graph:
            # loss
            show_history_1 = np.array(history)
            plt_1.plot(show_history_1[:,0:2])
            plt_1.legend(['Tr Loss', 'Val Loss'])
            plt_1.xlabel('Epoch Number')
            plt_1.ylabel('Loss')
            plt_1.ylim(0,1)
            plt_1.savefig(output_path+'/'+'loss_curve.png')
        epoch_end = time.time()
    
        print("Epoch : {:03d}, Training: Loss - {:.4f}, Accuracy - {:.4f}%, \n\t\tValidation : Loss - {:.4f}, Accuracy - {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        print("best_epoch: ", best_epoch)
        # Save if the model has best accuracy till now
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # saving model and checkpoint
        print("saving checkpoint of: ", str(epoch+1))
        torch.save({
            'epoch': str(epoch+1),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f"{output_path}/latest_ckpt.pt")
        torch.save(model,f"{output_path}/exported_latest_model.pt")