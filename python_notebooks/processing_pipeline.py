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

pdf_image = convert_from_path(pdf_path)
pages = list(pdf_image)

def opencv_conversion(pages):
    opencv_list_of_pages = []
    for page in pages:
        opencv_pageN = np.array(page)
        opencv_pageN = cv2.cvtColor(opencv_pageN, cv2.COLOR_RGB2BGR)
        opencv_list_of_pages.append(opencv_pageN)
    return opencv_list_of_pages

def apply_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray
def apply_gaussian_blur(image):      #smooths image uniformly
    gaussian = cv2.GaussianBlur(image,(3,3),0)
    return gaussian

def filter(image_array):
    filtered_image_list = []
    for page in image_array:
        