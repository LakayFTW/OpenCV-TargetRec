import cv2
import numpy as np
import PySimpleGUI as sg

# This code is completely generated by ChatGPT.
# This was generated to give an example on how to achieve the given goal

def detect_target(frame):
    # The input image is converted to greyscale mode to simplify processing.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Canny edge detection is applied to the greyscale image to extract the edges of the target.
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # The Hough transformation is used to recognise circles in the image. 
    # This involves searching for circles that could represent the shape of the archery target and its rings. 
    # The cv2.HoughCircles() function returns the recognised circles.
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=50, param2=30, minRadius=50, maxRadius=200)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # Sort the circles by radius
        circles = sorted(circles, key=lambda x: x[2], reverse=True)
        
        # Draw the largest circle (the target)
        x, y, r = circles[0]
        cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
        
        # Draw the inner rings (1-2: white, 3-4: black, 5-6: blue, 7-8: red, 9-10: yellow)
        colors = [(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
        for i, color in enumerate(colors, start=1):
            if i <= 10:
                cv2.circle(frame, (x, y), int(r * i / 10), color, 4)
    
    return frame

layout = [
	[sg.Image(filename="", key="image")],
]

window = sg.Window("Cam Viewer", layout, resizable=True)

# Main Program
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detected_frame = detect_target(frame)

    imgbytes = cv2.imencode(".png", detected_frame)[1].tobytes()

    event, values = window.read(timeout=20)
    if event == sg.WINDOW_CLOSED:
        break

    window["image"].update(data=imgbytes)

cap.release()
cv2.destroyAllWindows()
