import cv2 as cv
import numpy as np
import PySimpleGUI as sg

# Function to detect color
def detect_color(frame, color_ranges):
	# Convert the frame into hsv-spectrum
	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

	# Create empty mask
	mask = np.zeros(frame.shape[:2], dtype=np.uint8)
	
    # add the mask for every color
	for lower_bound, upper_bound in color_ranges:
		# Create maks for the color
		color_mask = cv.inRange(hsv, lower_bound, upper_bound)
	
		# Perform a morphological operation to smooth the mask
		color_mask = cv.erode(color_mask, None, iterations=2)
		color_mask = cv.dilate(color_mask, None, iterations=2)

		mask = cv.bitwise_or(mask, color_mask)

	colored_frame = cv.bitwise_and(frame, frame, mask=mask)

	# [b,g,r]
	colored_frame[mask == 0] = [0, 0, 0]
	
    # # Find contours in frame
	# contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
	
    # # If contours where found
	# if contours:
	# 	# Find die biggest contour
	# 	c = max(contours, key=cv.contourArea)
	# 	# Calculate the rectangle around the contour
	# 	x,y,w,h = cv.boundingRect(c)
	# 	# Draw the rectangle
	# 	cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)

	return colored_frame

rgb_color_blue = np.uint8([[[1, 117, 220]]])
rgb_color_red = np.uint8([[[189, 68, 42]]])
rgb_color_yellow = np.uint8([[[187, 201, 47]]])
rgb_color_black = np.uint8([[[0, 0, 0]]])
rgb_color_white = np.uint8([[[255, 255, 255]]])

hsv_color_blue = cv.cvtColor(rgb_color_blue, cv.COLOR_RGB2HSV)
hsv_color_red = cv.cvtColor(rgb_color_red, cv.COLOR_RGB2HSV)
hsv_color_yellow = cv.cvtColor(rgb_color_yellow, cv.COLOR_RGB2HSV)
# hsv_color_black = cv.cvtColor(rgb_color_black, cv.COLOR_RGB2HSV)
# hsv_color_white = cv.cvtColor(rgb_color_white, cv.COLOR_RGB2HSV)

tolerance = 40

# Define the range of the color you want to detect (here: blue) [tone, saturation, value]
lower_blue = np.array([hsv_color_blue[0][0][0] - tolerance, 100, 100])
upper_blue = np.array([hsv_color_blue[0][0][0] + tolerance, 255, 255])

lower_red = np.array([hsv_color_red[0][0][0] - tolerance, 100, 100])
upper_red = np.array([hsv_color_red[0][0][0] + tolerance, 255, 255])

lower_yellow = np.array([hsv_color_yellow[0][0][0] - tolerance, 100, 100])
upper_yellow = np.array([hsv_color_yellow[0][0][0] + tolerance, 255, 255])

# lower_black = np.array([hsv_color_black[0][0][0] - 0,0,0])
# upper_black = np.array([hsv_color_black[0][0][0] - 0,0,0])

# lower_white = np.array([hsv_color_white[0][0][0] - 0,0,0])
# upper_white = np.array([hsv_color_white[0][0][0] - 0,0,0])



color_ranges = [
	(lower_blue, upper_blue),
	(lower_red, upper_red),
	(lower_yellow, upper_yellow),
	# (lower_black, upper_black),
	# (lower_white, upper_white)
]

layout = [
	[sg.Image(filename="", key="image")],
]

window = sg.Window("Cam Viewer", layout, resizable=True)

# Capture frames from a camera with device index=0
cap = cv.VideoCapture(0)

# loop runs if capturing has been initialized 
while True: 

	# reads frame from a camera 
	ret, frame = cap.read()
	if not ret:
		break
 
	# execute color recognition
	frame = detect_color(frame, color_ranges)

	imgbytes = cv.imencode(".png", frame)[1].tobytes()

	# Check for events
	event, values = window.read(timeout=20)
	if event == sg.WINDOW_CLOSED:
		break

	# Update the image in the window
	window["image"].update(data=imgbytes)

# release the camera from video capture
cap.release() 

# De-allocate any associated memory usage 
cv.destroyAllWindows() 