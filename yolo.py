import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# Load the YOLOv8 model for vehicle type recognition
vehicle_model = YOLO('yolov8n.pt')

# Load the image
img = cv2.imread("D:\\Downloads\\240416114616-Vehicle.jpg")

# Perform vehicle recognition using YOLOv8
results = vehicle_model(img)

# Loop through the detected vehicles
for detection in results:
    # Get the bounding box coordinates
    boxes = detection.boxes.xyxy

    # Loop through the bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()  # Convert the tensor to a list

        # Extract the vehicle region
        vehicle_region = img[int(y1):int(y2), int(x1):int(x2)]

        #... rest of your code...

        # Convert the vehicle region to grayscale
        gray_vehicle = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to segment the license plate
        _, thresh = cv2.threshold(gray_vehicle, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through the contours
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            if area > 100 and aspect_ratio > 2:
                # Draw a rectangle around the license plate
                cv2.rectangle(vehicle_region, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Extract the license plate text using EasyOCR
                reader = easyocr.Reader(['en'])
                result = reader.readtext(vehicle_region)

                # Print the vehicle type and license plate number
                print("Vehicle Type:", result[0][1])
                print("License Plate Number:", result[1][1])

# Display the image with the detected vehicles and license plates
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()