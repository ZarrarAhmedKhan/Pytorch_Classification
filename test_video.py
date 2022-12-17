'''
RUN: python3 test_video.py video.mp4
'''
import sys
from PIL import Image
import torch, torchvision
from torchvision import  transforms
import numpy as np
# import matplotlib.pyplot as plt
import cv2
from time import time

# image_transforms = {
#     'test': transforms.Compose([
#         transforms.ToTensor()]
#         )
# }
image_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size=[640,480]),
        # transforms.CenterCrop(size=448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
idx_to_class = {0: 'fake', 1: 'real'}
model = torch.load('08-16-PM_model_49_new.pt', map_location=torch.device('cpu'))
model.eval()

def transform_frame(test_image):
    transform = image_transforms['test']
    print("test_image: ", test_image.shape)
    # test_image = cv2.resize(test_image, (1280,720))
    color_converted = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image=Image.fromarray(color_converted)
    # test_image = Image.open(test_image_name)
    # plt.imshow(test_image)
    # print("yes")  
    test_image_tensor = transform(test_image)
    return test_image_tensor

def predict(test_image):

    # if torch.cuda.is_available():
    #     test_image_tensor = test_image_tensor.view(1, 3, 1280, 720).cuda()
    # else:

    test_image_tensor = transform_frame(test_image)
    test_image_tensor = test_image_tensor.view(1, 3, 448, 448)
    
    with torch.no_grad():
        out = model(test_image_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(1, dim=1)
        clas = idx_to_class[topclass.cpu().numpy()[0][0]]
        score = topk.cpu().numpy()[0][0]
        print("cls: ", clas)
    return clas, score

if __name__ == '__main__':
    vid = cv2.VideoCapture(sys.argv[1])

    while True:
        start_time = time()
        ret, frame = vid.read()
        if not ret:
            break
        clas, score = predict(frame)
        score = round(score, 2)
        print("score: ", score)
        end_time = time()
        seconds = end_time - start_time
        fps  = round(1 / seconds,2)
        print("Estimated frames per second : {0}".format(fps))
        frame = cv2.putText(frame, f'{clas} score: {score} fps: {fps}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1,cv2.LINE_AA, False)
        cv2.imshow("vid", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

