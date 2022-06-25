from keras.models import load_model
import cv2
import numpy as np 
import copy

face_classifier = cv2.CascadeClassifier('./haar_face.xml')
classifier = load_model('./model.h5')

emo_labels  = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Suprise']

cap = cv2.VideoCapture(0)

while True:
    
    _, frame = cap.read()
    img = copy.deepcopy(frame)
    img = cv2.flip(frame,1) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        roi = cv2.resize(roi_gray, (48,48))
        prediction = classifier.predict(roi[np.newaxis, :, :, np.newaxis])

        label = emo_labels[np.argmax(prediction)]
        
        cv2.putText(img,label,(x, y-5),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)

        cv2.putText(img,'Angry: '+str(prediction[0][0]),(0, 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
        cv2.putText(img,'Disgust: '+str(prediction[0][1]),(0, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
        cv2.putText(img,'Fear: '+str(prediction[0][0]),(0, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
        cv2.putText(img,'Happy: '+str(prediction[0][0]),(0, 55),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
        cv2.putText(img,'Neutral: '+str(prediction[0][0]),(0, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
        cv2.putText(img,'Sad: '+str(prediction[0][0]),(0, 85),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
        cv2.putText(img,'Suprise: '+str(prediction[0][0]),(0, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)

        img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255),2)
            
    
    cv2.imshow("Emotion Dectector", img)
    key = cv2.waitKey(1) & 0xFF
    if key== ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
        