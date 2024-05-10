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

rgb_color_red = np.uint8([[[227, 49, 40]]])
rgb_color_lightblue = np.uint8([[[1, 117, 220]]])
rgb_color_black = np.uint8([[[27, 25, 26]]])
rgb_color_white = np.uint8([[[251, 251, 251]]])
rgb_color_yellow = np.uint8([[[253, 233, 73]]])

hsv_color_red = cv.cvtColor(rgb_color_red, cv.COLOR_RGB2HSV)
hsv_color_lightblue = cv.cvtColor(rgb_color_lightblue, cv.COLOR_RGB2HSV)
hsv_color_black = cv.cvtColor(rgb_color_black, cv.COLOR_BGR2HSV)
hsv_color_white = cv.cvtColor(rgb_color_white, cv.COLOR_BGR2HSV)
hsv_color_yellow = cv.cvtColor(rgb_color_yellow, cv.COLOR_BGR2HSV)

tolerance = 20

# Define the range of the colour you want to detect (here: blue) [tone, saturation, value]
lower_blue = np.array([hsv_color_lightblue[0][0][0] - tolerance, 100, 100])
upper_blue = np.array([hsv_color_lightblue[0][0][0] + tolerance, 255, 255])

# Define the range of red
lower_red = np.array([hsv_color_red[0][0][0] - tolerance, 100, 100])
upper_red = np.array([hsv_color_red[0][0][0] + tolerance, 255, 255])

lower_yellow = np.array([hsv_color_yellow[0][0][0] - tolerance, 100, 100])
upper_yellow = np.array([hsv_color_yellow[0][0][0] + tolerance, 255, 255])

lower_black = np.array([hsv_color_black[0][0][0] - tolerance, 100, 100])
upper_black = np.array([hsv_color_black[0][0][0] + tolerance, 255, 255])

lower_white = np.array([hsv_color_white[0][0][0] - tolerance, 100, 100])
upper_white = np.array([hsv_color_white[0][0][0] + tolerance, 255, 255])


color_ranges = [
	(lower_blue, upper_blue),
	(lower_red, upper_red),
	(lower_black, upper_black),
	(lower_white, upper_white),
	(lower_yellow, upper_yellow)
]

layout = [
	[sg.Image(filename="", key="image")],
	[sg.HSeparator()],
	[sg.Text("Color Percentage")],
    [sg.Text("Blue:"), sg.Text("0%", key="blue_percent")],
    [sg.Text("Red:"), sg.Text("0%", key="red_percent")],
    [sg.Text("Black:"), sg.Text("0%", key="black_percent")],
    [sg.Text("White:"), sg.Text("0%", key="white_percent")],
    [sg.Text("Yellow:"), sg.Text("0%", key="yellow_percent")]
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

	# Calculate color percentages
	total_pixels = np.sum(frame > 0)
	blue_pixels = np.sum(frame == hsv_color_lightblue)
	red_pixels = np.sum(frame == hsv_color_red)
	white_pixels = np.sum(frame == hsv_color_white)
	yellow_pixels = np.sum(frame == hsv_color_yellow)
	black_pixels = total_pixels - blue_pixels - red_pixels - white_pixels - yellow_pixels

	# Update color percentage text elements
	window["blue_percent"].update(f"{(blue_pixels / total_pixels) * 100:.2f}%")
	window["red_percent"].update(f"{(red_pixels / total_pixels) * 100:.2f}%")
	window["black_percent"].update(f"{(black_pixels / total_pixels) * 100:.2f}%")
	window["white_percent"].update(f"{(white_pixels / total_pixels) * 100:.2f}%")
	window["yellow_percent"].update(f"{(yellow_pixels / total_pixels) * 100:.2f}%")

	# Update the image in the window
	window["image"].update(data=imgbytes)

# release the camera from video capture
cap.release() 

# De-allocate any associated memory usage 
cv.destroyAllWindows() 