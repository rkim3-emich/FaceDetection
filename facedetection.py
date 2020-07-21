import cv2
from math import sqrt

#Get face identifying classifier
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#Get webcam
webcam = cv2.VideoCapture(0)


#Exits internally
while True:
    #Get frame from webcam and flip frame so it isn't mirrored
    _, frame = webcam.read()
    frame = cv2.flip(frame, 1)

    #Convert to frame to grayscale for interpretation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Find faces
    faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    identified = frame.copy()

    #Iterate through faces in image and draw black circle over them
    for (x, y, w, h) in faces:
        radius = int((sqrt((w)**2+(h)**2)) // 2)
        cv2.circle(identified, (x+w//2, y+h//2), radius, (0, 0, 0), -1)

    #Display original and edited video feeds
    cv2.imshow("Original", frame)
    cv2.imshow("Face Covered", identified)

    #Exit loop if user eneters q and has numlock on
    if cv2.waitKey(1) & 0b11111111 == ord('q'):
        break

#Clean up
webcam.release()
cv2.destroyAllWindows()
