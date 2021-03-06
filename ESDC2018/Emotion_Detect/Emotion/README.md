# Notice
This folder contains the orginal code we referred to ('emotions.py') and the modified version ('FaceAndEmotionDetection_v2.py') that we used for the contest. The following sections were copied from the original readme file.

In './models/' folder, 'emotion_model.hdf5' were for the original code while 'new_emotion_train_v1.h5' were copied and renamed from the training folder.

If you want to analyze some video clip, put it into './demo' folder and name it as 'test.mp4'. You can also use a camera to get real time response.
# Emotion
This software recognizes human faces and their corresponding emotions from a video or webcam feed. Powered by OpenCV and Deep Learning.

![Demo](https://github.com/petercunha/Emotion/blob/master/demo/demo.gif?raw=true)

## Deep Learning Model

The model used is from this [research paper](https://github.com/oarriaga/face_classification/blob/master/report.pdf) written by Octavio Arriaga, Paul G. Plöger, and Matias Valdenegro.

![Model](https://i.imgur.com/vr9yDaF.png?1)

## Credit

* Computer vision powered by OpenCV.
* Neural network scaffolding powered by Keras with Tensorflow.
* Convolutional Neural Network (CNN) deep learning architecture is from this [research paper](https://github.com/oarriaga/face_classification/blob/master/report.pdf).
* Pretrained Keras model and much of the OpenCV code provided by GitHub user [oarriaga](https://github.com/oarriaga).
