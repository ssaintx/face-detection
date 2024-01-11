import cv2

face_cascade = cv2.CascadeClassifier('model/model.xml') # load trained model
image = cv2.imread('test/test.jpg') # path to image

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # making it gray

faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5) # detect faces

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2) # draw rectangle to face

cv2.imshow('Face Recognition', image) # show the result
cv2.waitKey(0)
cv2.destroyAllWindows()