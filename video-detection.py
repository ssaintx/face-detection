import cv2

face_cascade = cv2.CascadeClassifier('model/model.xml') # trained model
video_capture = cv2.VideoCapture(0) # turning on web cam

while True:
    ret, frame = video_capture.read() # read our pic

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # setting it gray

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5) # detect faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # draw a rectangle to face

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # quit program when we type 'q'
        break

video_capture.release()
cv2.destroyAllWindows()