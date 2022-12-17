'''
RUN:
    *IMAGES FOLDER* 
    python3 test_image.py test_folder output_folder
    
    *SINGLE IMAGE*
    python3 test_image.py image.jpg
'''
import sys
from PIL import Image
import torch, torchvision
import torch.nn as nn
from torchvision import  transforms,models
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from glob import glob

image_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size=[640,468]),
        # transforms.CenterCrop(size=448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
# image_transforms = {
#     'test': transforms.Compose([
#         transforms.ToTensor()
#     ])
# }

idx_to_class = {0: 'fake', 1: 'real'}
model = torch.load('pretrained_models/exported_latest_model.pt', map_location=torch.device('cpu'))
model.eval()

def predict(test_image, visualize = True):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    '''
    orig_image = test_image.copy()
    transform = image_transforms['test']

    # test_image = cv2.resize(test_image, (448,448))
    color_converted = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image=Image.fromarray(color_converted)

    # test_image = Image.open(test_image_name)
    # plt.imshow(test_image)
    # print("yes")  
    test_image_tensor = transform(test_image)
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 1280, 720).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 640, 468)

    print("test_image_tensor: ", test_image_tensor.shape)
    with torch.no_grad():
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        # print(ps.topk(2, dim=1))
        topk, topclass = ps.topk(1, dim=1)
        clas = idx_to_class[topclass.cpu().numpy()[0][0]]
        score = topk.cpu().numpy()[0][0]
        score = round(score, 3)
        print("clas: ", clas)

        for i in range(1):
            print("Predcition", i+1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])

    if visualize:
        orig_image = cv2.resize(orig_image, (480,640))
        orig_image = cv2.putText(orig_image, f'{clas} score: {score}', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1,cv2.LINE_AA, False)
        cv2.imshow("image", orig_image)
        cv2.waitKey(0)
    return clas

if __name__ == '__main__':

    name = sys.argv[1]
    if os.path.isdir(name):
        output_folder = sys.argv[2]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print("folder")
        for image in glob(f'{name}/*'):
            print(image)
            image_name = image.split('/')[-1]
            frame = cv2.imread(image)
            class_name = predict(frame, visualize = False)
            output_path = os.path.join(output_folder, class_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_file_path = os.path.join(output_path, image_name)
            if os.path.exists(output_file_path):
                continue
            cv2.imwrite(f'{output_file_path}', frame)

    else:
        print("file")
        image = cv2.imread(name)
        predict(image)