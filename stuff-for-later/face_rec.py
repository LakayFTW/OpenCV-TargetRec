import cv2 as cv

# load the trained hair-based face recognitionmodel 
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture frames from a camera with device index=0
cap = cv.VideoCapture(0)

# loop runs if capturing has been initialized 
while(1): 

	# reads frame from a camera 
	ret, frame = cap.read() 
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	
    # Recognize faces in frame
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
	
    # draw rectangle around face
	for (x, y, w, h) in faces:
		cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

	# Display the frame
	cv.imshow('Camera',frame) 
	
	# Wait for 25ms
	if cv.waitKey(1) & 0xFF == ord('q'):
		break
		
# release the camera from video capture
cap.release() 

# De-allocate any associated memory usage 
cv.destroyAllWindows() 