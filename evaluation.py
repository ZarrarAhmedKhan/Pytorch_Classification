import torch
import torch.nn as nn
import torch, torchvision
from torchvision import models
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import sys
# from torch import ConfusionMatrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
from time import time
import os

image_transforms = {
    'test': transforms.Compose([
        transforms.RandomResizedCrop(size=[640,480], scale=(0.8, 1.0)),
        # transforms.RandomRotation(degrees=15),
        # transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(size=448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
    
    def my_model(self, num_classes):
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

def computeTestSetAccuracy(model, loss_criterion, test_data_loader, test_data_size):
    '''
    Function to compute the accuracy on the test set
    Parameters
        :param model: Model to test
        :param loss_criterion: Loss Criterion to minimize
    '''

    test_acc = 0.0
    test_loss = 0.0

    y_pred = []
    y_true = []

    # Validation - No gradient tracking needed
    with torch.no_grad():

        # Set to evaluation mode
        model.eval()

        # Validation loop
        for j, (inputs, labels) in enumerate(test_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print("labels: ", labels)


            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            # print("output: ", outputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Compute the total loss for the batch and add it to valid_loss
            test_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)

            pred = (predictions).data.cpu().numpy()
            y_pred.extend(pred) # Save Prediction

            label = labels.data.cpu().numpy()
            # print("labels: ", label)
            # labels
            y_true.extend(label) # Save Truth

            # print("predictions: ", pred)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            test_acc += acc.item() * inputs.size(0)

            print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

    # Find average test loss and test accuracy
    avg_test_loss = test_loss/test_data_size 
    avg_test_acc = test_acc/test_data_size

    print("Test accuracy : " + str(avg_test_acc))
    print("Test loss : " + str(avg_test_loss))

    return y_pred, y_true

def evaluate(test_data, model_path, output_path):

    batch_size = 1
    test_directory = test_data
    # Number of classes
    num_classes = len(os.listdir(test_directory))  #10#2#257
    print("num_classes: ", num_classes)
    # Load Data from folders
    data = {
        'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
    }
    idx_to_class = {v: k for k, v in data['test'].class_to_idx.items()}
    print(idx_to_class)
    classes_names = list(idx_to_class.values())
    print("classes_names: ", classes_names)
    # exit(0)

    test_data_size = len(data['test'])
    print("test_data_size: ", test_data_size)

    # Create iterators for the Data loaded using DataLoader module
    test_data_loader = DataLoader(data['test'], batch_size=batch_size, shuffle=True)

    net= Classifier()
    model = net.my_model(num_classes)
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    loss_criterion = nn.NLLLoss()

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    y_pred, y_true = computeTestSetAccuracy(model, loss_criterion, test_data_loader, test_data_size)

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *len(classes_names), index = [i for i in classes_names],
                        columns = [i for i in classes_names])
    
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f'{output_path}_matrix.png')

    y_true_tensor = torch.Tensor(y_true)
    y_pred_tensor = torch.Tensor(y_pred)

    # precision and recall report
    print(classification_report(y_true_tensor,y_pred_tensor))

if __name__ == '__main__':
    test_data = sys.argv[1]
    # path_to_latest_checkpoint
    model_path = sys.argv[2]
    output_path = model_path.split('/')[:-1]
    output_path = '/'.join(output_path)
    test_data_name = test_data.split('/')[-1]
    output_path = output_path + '/' + test_data_name
    print(output_path)
    evaluate(test_data, model_path, output_path)