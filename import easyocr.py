import easyocr
import pandas as pd
import matplotlib.pyplot as plt
import cv2
image_path = "D:\\Downloads\\240416140204-Vehicle.jpg"
img = cv2.imread(image_path)
reader = easyocr.Reader(['en'], gpu=True)

results = reader.readtext(img)

df = pd.DataFrame(results, columns=['bbox', 'text', 'conf'])
print(df)
for result in results:
    bbox, text, conf = result
    x0, y0 = bbox[0]
    x1, y1 = bbox[2]
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(img, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()