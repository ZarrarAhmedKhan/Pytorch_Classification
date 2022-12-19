# Pytorch_Classification

*** RUN ***

## Training (Transfer Learning)

* set dataset_path (containing images folders of all classes)

* set output_path

* set batch_size

> `!python3 train_classifier.py`

## ReTraining

* set restored_path also

## Evaluation 

> `!python3 evaluation.py test_data_path model_path`

* test_data_path contain --> ('real' and 'fake') all classes folder

## Testing

*IMAGES FOLDER*

> `python3 test_image.py test_folder output_folder`
    
*SINGLE IMAGE*

> `!python3 test_image.py image.jpg`

*VIDEO*

> `!python3 test_video.py video.mp4`
