import os

from classifier import train
from classifier import preprocessing

def main():
    '''
        INPUTS
        
        models: 'resnet50'
        dataset contain two folders "train" and "test"
    '''
    dataset = 'face_dataset'
    pretrained_model = 'resnet50'
    # Batch size
    bs = 32
    output_path = os.path.join('train', pretrained_model)
    num_epochs = 50

    print("pretrained_model: ", pretrained_model)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # prprocessing
    paras = preprocessing(dataset, bs)

    # train
    train(num_epochs, paras, output_path=output_path)

if __name__ == '__main__':
    main()