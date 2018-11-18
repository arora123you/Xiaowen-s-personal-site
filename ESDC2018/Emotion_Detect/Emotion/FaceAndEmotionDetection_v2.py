import sys
import os
reload(sys)
sys.setdefaultencoding('utf8')
import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import face_recognition
import time
import csv
USE_WEBCAM = False # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './models/new_emotion_train_v1.h5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_list = ['suprise','fear','sad','disgust','happy','angry']
# starting lists for calculating modes
emotion_window = []
face_names = []
# starting video streaming

class get_known_people():
    
    def __init__(self,path):
        now = os.getcwd()
        now = now+path
        self.n = os.listdir(now)
        self.path = now
    
    #list real_name : name with 'xxx.jpg' 
    #list real_n : just the name
    def name(self):
        name = self.n
        real_name =[];real_n=[]
        for i in range(len(name)):
            if (".jpg" in name[i]) or (".png" in name[i]) or ( ".jpeg" in name[i]) or ( ".JPEG" in name[i]) or ( ".PNG" in name[i]) or ( ".JPG" in name[i]):
                real_name.append(name[i])
                real_n.append(name[i].split('.')[0])
        self.real_n = real_name
        return [real_n,self.real_n]

    # encoding pics in the list encoding_name
    def encoding_name(self,path):
        name = self.real_n
        encoding_name =[]
        for i in range(len(self.real_n)):
            tmp = '.'+path+name[i]
            tmp = face_recognition.load_image_file(tmp)
            tmp = face_recognition.face_encodings(tmp)[0]
            encoding_name.append(tmp)
        return encoding_name


def who_are_you(known_name,path,known_path):
    a=get_known_people(path)
    name=a.name()[1]
    unknown_face_encoding=a.encoding_name(path)
    for i in range(len(unknown_face_encoding)):
        results = face_recognition.compare_faces(known_faces, unknown_face_encoding[i])
        if True in results:
            num = results.index(True)
            print('You are',known_name[num])
        else :
            print("Sorry I don't know who you are. Would you like to be added into the database?")
            im=Image.open('.'+path+name[i])
            im.show()
            your_name=raw_input("Please input your name")+'.png'
            os.rename('.'+path+name[i],your_name)
            im.save(os.path.join('.'+known_path,os.path.basename('.'+path+your_name)))


path="/known_people/"
a=get_known_people(path)
name=a.name()[0]
print(name)
known_faces=a.encoding_name(path)

#path2='/unknown_people/'
#who_are_you(name,path2,path)

known_face_encodings=known_faces
known_face_names=name

cv2.namedWindow('window_frame')

if (USE_WEBCAM == True):
    
    video_capture = cv2.VideoCapture(0) # Webcam source
else:
    video_capture = cv2.VideoCapture('./demo/test.mp4') # Video file source
    print(video_capture)

# fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
# size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# out = cv2.VideoWriter('./videos/002_output.avi',fourcc, 20.0, size, True)
add_unknown_face_mode = False
while video_capture.isOpened(): # True:
    ret, bgr_image = video_capture.read()

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
            minSize=(50,50), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        rgb_face = rgb_image[y1:y2, x1:x2]
        #face_recoginiton
        face_location = [(y1,x2,y2,x1)]
        #print(face_location)
        face_encoding = face_recognition.face_encodings(rgb_image, face_location)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0])
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            #print(time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())))
        elif add_unknown_face_mode:
            print("Do you want to be added into the database? press 'y' or 'n' to continue\n")
            if cv2.waitKey(0) & 0xFF == ord("n"):
                break
            if cv2.waitKey(0) & 0xFF == ord("y"):
                your_name='.'+path+raw_input("Please input your name\n")+'.png'
                cv2.imwrite(your_name, bgr_image)
                path="/known_people/"
                a=get_known_people(path)
                names=a.name()[0]
                known_faces=a.encoding_name(path)
                known_face_encodings=known_faces
                known_face_names=names
        #If a match was found in known_face_encodings, just use the first one.
        face_names.append(name)
        try:
            #gray_face = cv2.resize(gray_face, (emotion_target_size))
            rgb_face = cv2.resize(rgb_face, (emotion_target_size))
        except:
            continue
        rgb_face = preprocess_input(rgb_face)
        #gray_face = np.expand_dims(gray_face, 0)
        rgb_face = np.expand_dims(rgb_face, 0)
        emotion_prediction = emotion_classifier.predict(rgb_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,face_names[-1],
                  color, 0, -45, 1, 1)
        if name != "Unknown":
            if not os.path.exists("./known_people_emotions/"+name):
                os.makedirs("./known_people_emotions/"+name)
                with open("./known_people_emotions/"+name+"/"+name+".csv", 'w') as csvfile:
                    fieldnames = ['name', 'time','emotion','emotion_vector']
                    writer = csv.writer(csvfile)
                    writer.writerow(fieldnames)
            emotion_vector = [0,0,0,0,0,0]
            for i in range(len(emotion_list)):
                if emotion_mode == emotion_list[i]:
                    emotion_vector[i] += 1
                    break
            with open("./known_people_emotions/"+name+"/"+name+".csv", 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([name,time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())),emotion_mode,emotion_vector])
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("NOOOOOOOOO")
        break
#out.release()
video_capture.release()
cv2.destroyAllWindows()
