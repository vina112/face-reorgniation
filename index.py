import cv2,time
import pickle
import numpy as np

face_cascade = cv2.CascadeClassifier("data\haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels={"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

video = cv2.VideoCapture(0)

a = 1

while True:
    a=a+1
    check,frame = video.read()
    #print(frame)
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors = 5)

    for (x,y,w,h) in faces:
        #print(x,y,w,h) it would have printed the location co-ordinates if image
        cv2.rectangle(frame,(x,y),(x+w , y+h),(100,125,223),3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)

        if conf>=45 and conf<=80:
            #print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)



        img_item = "my-img.png"
        cv2.imwrite(img_item,roi_gray)
        eyes = eye_cascade.detectMultiScale(roi_gray,1.05,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

    cv2.imshow("video",frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
print("frames: ",a)
video.release()
cv2.destroyAllWindows()

