import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import re

# read image
image_path = r"D:\Downloads\240416114616-Vehicle.jpg"

img = cv2.imread(image_path)

# instance text detector
reader = easyocr.Reader(['en'], gpu=False)

# detect text on image
text_ = reader.readtext(img)

threshold = 0.25
# draw bbox and text
for t_, t in enumerate(text_):
    print(t)

    bbox, text, score = t

    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
img = cv2.imread(image_path)

# instance text detector
reader = easyocr.Reader(['en'], gpu=False)

# detect text on image
text_ = reader.readtext(img)

threshold = 0.25
number_plate = ""

# draw bbox and text
for t_, t in enumerate(text_):
    print(t)

    bbox, text, score = t

    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

        # Filter number plate (assuming it's a plate number with 7-10 characters, and contains at least one digit and one uppercase letter)
        if len(text) >= 7 and len(text) <= 10 and any(char.isdigit() for char in text) and any(char.isupper() for char in text):
            number_plate = text

# Improve number plate detection using regular expression
pattern = re.compile(r'[A-Z]{2,3}[0-9]{1,4}[A-Z]{1,2}')
match = pattern.search(number_plate)
if match:
    number_plate = match.group()

print("Number Plate:", number_plate)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()