#This file will: 
# 0) Do imports and define functions
# 1) load a pdf
# 2) convert pdf to numpy array
# 3) apply grayscale and gaussian blur
# 4) Take the image from 3 and apply bounding boxes

# Each step may have a specific function. 1 or 2 functions should encapsulate this whole procedure

import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    list_of_pages = convert_from_path(pdf_path)

    #This converts the pdf pages into opencv arrays stored in a list
    opencv_pages = []
    for i in range(1,2):
        opencv_pageN = np.array(list_of_pages[i])
        opencv_pageN = cv2.cvtColor(opencv_pageN, cv2.COLOR_RGB2BGR)
        opencv_pages.append(opencv_pageN)

    #This applies a grayscale and gaussian blur to each page of the pdf. Stores them in a new list
    for i in range(len(opencv_pages)):
        gray = cv2.cvtColor(opencv_pages[i], cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray,(3,3),0)
        opencv_pages[i] = gaussian
    return opencv_pages

#Creating bounding boxes______________________________________________________________________
def word_boxes(filtered_image_list):
    for page in filtered_image_list:
        ocr_image = page.copy()
        ocr_data = pytesseract.image_to_data(ocr_image, output_type=pytesseract.Output.DICT)
        # Create a pandas DataFrame from OCR results
        df_ocr = pd.DataFrame(ocr_data)
    return df_ocr

#The above code creates bounding boxes for each image. This data needs to be fed to connor's portion
#where they will aggregate per line.

def cleaning_data(df): #this function will clean up the data by sorting the boxes top values and left values in increasing order and removes any duplicates

    df_NeededData = df[['left','top','width','height']].copy()

    #make sure all of the numbers are actually numbers and not strings 
    df_NeededData['top'] = pd.to_numeric(df_NeededData['top'])
    df_NeededData['left'] = pd.to_numeric(df_NeededData['left'])
    df_NeededData['width'] = pd.to_numeric(df_NeededData['width'])
    df_NeededData['height'] = pd.to_numeric(df_NeededData['height'])


    df_NeededData = df_NeededData.sort_values(by=['top', 'left']).reset_index(drop=True) #sorts the data from top to bottom, left to right

    line_threshold_top = 15  # pixel tolerance for same line
    for i in range(0,len(df_NeededData) - 1):
        if(abs( df_NeededData.loc[i, 'top'] - df_NeededData.loc[i+1,'top']) < line_threshold_top): #checks to see if 2 top coords in a row are similar
            df_NeededData.loc[i+1, 'top'] = df_NeededData.loc[i, 'top'] #set the values equal

    df_NeededData = df_NeededData.sort_values(by=['top', 'left']).reset_index(drop=True) #sorts the data from top to bottom, left to right

    line_threshold_left = 15  # pixel tolerance for same line
    for i in range(0,len(df_NeededData) - 1):
        if(abs( df_NeededData.loc[i, 'left'] - df_NeededData.loc[i+1,'left']) < line_threshold_left): #checks to see if 2 top coords in a row are similar
            df_NeededData.loc[i+1, 'left'] = df_NeededData.loc[i, 'left'] #set the values equal


    df_NeededData = df_NeededData.drop_duplicates(subset=['left', 'top', 'width', 'height']).reset_index(drop=True) #deletes all duplicates and resets the index
    #even after everything above some of the bounding boxes are overlapping, so we just want to keep the largest box 
    #below is the code for how to only keep the biggest bounding box and remove the others
    df_NeededData['area'] = df_NeededData['width'] * df_NeededData['height'] # Compute area of each box
    df_NeededData = df_NeededData.sort_values('area', ascending=False) # Sort by area descending so largest box is first within each group
    df_NeededData = df_NeededData.drop_duplicates(subset=['left', 'top'], keep='first')# Drop duplicates keeping largest area box per (left, top)
    df_NeededData = df_NeededData.drop(columns=['area']).reset_index(drop=True) # Optionally, drop the area column afterward


    df_NeededData = df_NeededData.sort_values(by=['top', 'left']).reset_index(drop=True) #sorts the data from top to bottom, left to right

    return df_NeededData #this is the cleaned data 

def word_boxes_to_sentence_boxes(df_NeededData):#argument is the cleaned/sorted data
    #this function will take the individual bounding boxes around each word and combine them into boudning boxes for each line
    
    horizontal_threshold = 20

    resetLine = 0 # restart/save variable
    wordIndex = 1
    lineIndex = 1

    df_NeededData['line #'] = 0
    df_NeededData['word #'] = 0

    #_______________________________________________________________________________________
    line_boxes = []
    for top_val, group in df_NeededData.groupby('top'):
        left = group['left'].min()
        right = (group['left'] + group['width']).max()
        top = group['top'].min()
        bottom = (group['top'] + group['height']).max()
        width = right - left
        height = bottom - top
        line_boxes.append({'left': left, 'top': top, 'width': width, 'height': height})


    #___________________________________________________________________________________________

    for i in range(0,len(df_NeededData)):

        if i == resetLine:
            RP = df_NeededData.iloc[i]['left'] + df_NeededData.iloc[i]['width']
            df_NeededData.loc[i, 'line #'] = lineIndex
            df_NeededData.loc[i, 'word #'] = wordIndex
            lineIndex += 1 #if we hit a new line we want to index the line
            wordIndex += 1 #if we assign a new line we need to index the word number
            #print(f"Line has been reset and the new line is: {lineIndex}")
            continue
        
        else:
            LP = df_NeededData.iloc[i]['left']
            if(abs(LP - RP) < horizontal_threshold):
                df_NeededData.loc[i, 'line #'] = lineIndex
                df_NeededData.loc[i, 'word #'] = wordIndex
                wordIndex += 1 #if the words are on the same line them we want to keep indexing the words
                #print(f"Less than horizontal_threshold: {abs(LP-RP)}")
            else:
                resetLine = i+1 #move to the next line 
                wordIndex = 1 #reset the word index if we move to a new line
                #print(f"A new line needs to be reset. i = {i}, resetLine = {resetLine}, and wordIndex = {wordIndex}")
                
        RP = df_NeededData.iloc[i]['left'] + df_NeededData.iloc[i]['width']

    return df_NeededData, line_boxes #this returns the original cleaned/sorted data frame but with the line numbers and word numbers columns added to the end of it

def plot_page(png_file_path, line_boxes):#png_file_path needs to have quotations around it 

    ##I (Devin) added this df conversion bc I thought it was necessary
    line_boxes_df = pd.DataFrame(line_boxes)

    #PNG_NAME = 'output_page1.png'
    #PNG_NAME = png_file_path
    #img = cv2.imread(PNG_NAME)
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(png_file_path, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_rgb)

    for idx, row in line_boxes_df.iterrows():
        rect = patches.Rectangle(
            (row['left'], row['top']),
            row['width'],
            row['height'],
            linewidth=2,
            edgecolor='blue',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(row['left'], row['top']-5, f"Line {idx+1}", fontsize=8, color='blue', weight='bold')

    plt.axis('off')
    plt.show()
    return

def daddy(pdf_path):
    filtered_image_list = filter_pdf(pdf_path)
    pages_with_bounding_boxes = word_boxes(filtered_image_list)
    cleaned_and_sorted = cleaning_data(pages_with_bounding_boxes)
    #more_ cleaning contains sorted data with added columns for line#'s and word#'s
    more_cleaning, line_boxes = word_boxes_to_sentence_boxes(cleaned_and_sorted)
    plot_page(filtered_image_list[0], line_boxes)

    return

#Assign pdf_path, then apply the filters + bounding boxes
pdf_path = r"C:\Users\dkhun\UC Davis\AISC Github repository\BeginnerProjectFallQuarter2025\data\raw_pdfs\textbook_pdf_3_includes_diagrams.pdf"

df_ocr = daddy(pdf_path)

'''
THIS WAS VERIFICATION FOR THE BOUNDING BOXES
plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.title("Text Detection with Bounding Boxes\n(Green: High Confidence, Yellow: Medium, Red: Low)", 
          fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()

print("\nColor Legend:")
print("🟢 Green: Confidence > 80%")
print("🟡 Yellow: Confidence 60-80%")
print("🔴 Red: Confidence < 60%")
'''