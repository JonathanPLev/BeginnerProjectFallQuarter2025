#This file will: 
# 0) Do imports and define functions
# 1) load a pdf
# 2) convert pdf to numpy array
# 3) apply grayscale and gaussian blur
# 4) Take the image from 3 and apply bounding boxes

# Each step may have a specific function. 1 or 2 functions should encapsulate this whole procedure

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import PyPDF2
import pytesseract
from pdf2image import convert_from_path

pdf_path = r"C:\Users\dkhun\UC Davis\AISC Github repository\BeginnerProjectFallQuarter2025\data\raw_pdfs\textbook_pdf_3_includes_diagrams.pdf"

#This function takes a pdf path as an input. Its output is a list where each item is a filtered pdf page
def filter_pdf(pdf_path):
    pdf_image = convert_from_path(pdf_path)
    pages = list(pdf_image)

    #This converts the pdf pages into opencv arrays stored in a list
    opencv_list_of_pages = []
    for page in pages:
        opencv_pageN = np.array(page)
        opencv_pageN = cv2.cvtColor(opencv_pageN, cv2.COLOR_RGB2BGR)
        opencv_list_of_pages.append(opencv_pageN)

    #This applies a grayscale and gaussian blur to each page of the pdf. Stores them in a new list
    filtered_image_list = []
    for page in opencv_list_of_pages:
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray,(3,3),0)
        filtered_image_list.append(gaussian)
    return filtered_image_list

#Assign pdf_path, then apply the filters to it
filtered_image_list = filter_pdf(pdf_path)

#Creating bounding boxes______________________________________________________________________
for page in filtered_image_list:
    ocr_image = page.copy()
    ocr_data = pytesseract.image_to_data(ocr_image, output_type=pytesseract.Output.DICT)
    # Create a pandas DataFrame from OCR results
    df_ocr = pd.DataFrame(ocr_data)
    #print(df_ocr.head(10))
    # Filter out empty text
    df_ocr = df_ocr[df_ocr['text'].str.strip() != '']
    df_ocr = df_ocr[df_ocr['conf'] != -1]  # Remove invalid confidence scores

    image_with_boxes = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    #Draw bounding boxes for each word
    for idx, row in df_ocr.iterrows():
        x, y, w, h = row['left'], row['top'], row['width'], row['height']
        conf = row['conf']
        
        # Color based on confidence: green (high) to red (low)
        if conf > 80:
            color = (0, 255, 0)  # Green
        elif conf > 60:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
    
        # Draw rectangle
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
#The above code creates bounding boxes for each image. This data needs to be fed to connor's portion
#where they will aggregate per line.


plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
plt.title("Text Detection with Bounding Boxes\n(Green: High Confidence, Yellow: Medium, Red: Low)", 
          fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()

print("\nColor Legend:")
print("🟢 Green: Confidence > 80%")
print("🟡 Yellow: Confidence 60-80%")
print("🔴 Red: Confidence < 60%")