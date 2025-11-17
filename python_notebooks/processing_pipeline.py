# This file will:
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

HORIZONTAL_THRESHOLD = 15
LINE_THRESHOLD = 0
VERTICAL_THRESHOLD = 30
# i think there should be a check if ur on windows or on mac or linux.
pdf_path = "data/raw_pdfs/textbook_pdf_3_includes_diagrams.pdf"


# This function takes a pdf path as an input. Its output is a list where each item is a filtered pdf page
def filter_pdf(pdf_path):
    list_of_pages = convert_from_path(pdf_path)

    # This converts the pdf pages into opencv arrays stored in a list
    opencv_pages = []
    for i in range(1, 2):
        opencv_pageN = np.array(list_of_pages[i])
        opencv_pageN = cv2.cvtColor(opencv_pageN, cv2.COLOR_RGB2BGR)
        opencv_pages.append(opencv_pageN)

    # This applies a grayscale and gaussian blur to each page of the pdf. Stores them in a new list
    for i in range(len(opencv_pages)):
        gray = cv2.cvtColor(opencv_pages[i], cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
        opencv_pages[i] = gaussian
    return opencv_pages


# Creating bounding boxes______________________________________________________________________
def word_boxes(filtered_image_list):
    for page in filtered_image_list:
        ocr_image = page.copy()
        ocr_data = pytesseract.image_to_data(
            ocr_image, output_type=pytesseract.Output.DICT
        )
        # Create a pandas DataFrame from OCR results
        df_ocr = pd.DataFrame(ocr_data)

    return df_ocr


# The above code creates bounding boxes for each image. This data needs to be fed to connor's portion
# where they will aggregate per line.


def cleaning_data(
    df, filtered_image
):  # this function will clean up the data by sorting the boxes top values and left values in increasing order and removes any duplicates

    df_NeededData = df[["left", "top", "width", "height"]].copy()

    # make sure all of the numbers are actually numbers and not strings
    df_NeededData["top"] = pd.to_numeric(df_NeededData["top"])
    df_NeededData["left"] = pd.to_numeric(df_NeededData["left"])
    df_NeededData["width"] = pd.to_numeric(df_NeededData["width"])
    df_NeededData["height"] = pd.to_numeric(df_NeededData["height"])

    df_NeededData = df_NeededData.sort_values(by=["top", "left"]).reset_index(
        drop=True
    )  # sorts the data from top to bottom, left to right
    df_NeededData.to_csv("data/ocr_output/needed_data.csv", index=False)
    line_threshold_top = LINE_THRESHOLD  # pixel tolerance for same line
    for i in range(0, len(df_NeededData) - 1):
        if (
            abs(df_NeededData.loc[i, "top"] - df_NeededData.loc[i + 1, "top"])
            < line_threshold_top
        ):  # checks to see if 2 top coords in a row are similar
            df_NeededData.loc[i + 1, "top"] = df_NeededData.loc[
                i, "top"
            ]  # set the values equal

    df_NeededData = df_NeededData.sort_values(by=["top", "left"]).reset_index(
        drop=True
    )  # sorts the data from top to bottom, left to right
    df_NeededData.to_csv("data/ocr_output/sorted_data.csv", index=False)
    line_threshold_left = HORIZONTAL_THRESHOLD  # pixel tolerance for same line
    for i in range(0, len(df_NeededData) - 1):
        if (
            abs(df_NeededData.loc[i, "left"] - df_NeededData.loc[i + 1, "left"])
            < line_threshold_left
        ):  # checks to see if 2 left coords in a row are within threshold
            df_NeededData.loc[i + 1, "left"] = df_NeededData.loc[
                i, "left"
            ]  # set the values equal

    df_NeededData = df_NeededData.drop_duplicates(
        subset=["left", "top", "width", "height"]
    ).reset_index(
        drop=True
    )  # deletes all duplicates and resets the index

    # logic to remove bounding boxes that are too wide or tall
    df_NeededData = df_NeededData[
        (df_NeededData["width"] <= 200)  # Maximum width for a very long word
        & (df_NeededData["height"] <= 50)  # Maximum height for a line of text
    ].copy()
    # even after everything above some of the bounding boxes are overlapping, so we just want to keep the largest box
    # below is the code for how to only keep the biggest bounding box and remove the others
    df_NeededData["area"] = (
        df_NeededData["width"] * df_NeededData["height"]
    )  # Compute area of each box
    df_NeededData = df_NeededData.sort_values(
        "area", ascending=False
    )  # Sort by area descending so largest box is first within each group
    df_NeededData = df_NeededData.drop_duplicates(
        subset=["left", "top"], keep="first"
    )  # Drop duplicates keeping largest area box per (left, top)
    df_NeededData = df_NeededData.drop(columns=["area"]).reset_index(
        drop=True
    )  # Optionally, drop the area column afterward

    df_NeededData = df_NeededData.sort_values(by=["top", "left"]).reset_index(
        drop=True
    )  # sorts the data from top to bottom, left to right
    # plot_page(filtered_image, df_NeededData)
    return df_NeededData  # this is the cleaned data


def word_boxes_to_sentence_boxes(df):
    horizontal_threshold = HORIZONTAL_THRESHOLD
    vertical_threshold = VERTICAL_THRESHOLD

    # ---- SLOPE CORRECTION (fix for diagonal text) ----
    subset = df.iloc[: min(len(df), 20)]
    m = np.polyfit(subset["left"], subset["top"], 1)[0]  # slope estimate
    df["y_corr"] = df["top"] - m * df["left"]  # corrected y

    # Shifted comparisons
    left_shifted = df["left"].shift(1)
    width_shifted = df["width"].shift(1)
    y_shifted = df["y_corr"].shift(1)

    prev_right = left_shifted + width_shifted

    small_gap = (df["left"] - prev_right).abs() < horizontal_threshold
    same_vertical = (df["y_corr"] - y_shifted).abs() < vertical_threshold

    same_line = (same_vertical & small_gap).fillna(False)

    df["line #"] = (~same_line).cumsum()
    df["word #"] = df.groupby("line #").cumcount() + 1

    line_boxes = compute_line_boxes(df)
    return df, line_boxes


def compute_line_boxes(df):
    line_boxes = (
        df.groupby("line #")
        .apply(
            lambda g: pd.Series(
                {
                    "left": g["left"].min(),
                    "top": g["top"].min(),
                    "width": (g["left"] + g["width"]).max() - g["left"].min(),
                    "height": (g["top"] + g["height"]).max() - g["top"].min(),
                }
            )
        )
        .reset_index()
    )
    return line_boxes


def plot_page(
    png_file_path, line_boxes
):  # png_file_path needs to have quotations around it

    ##I (Devin) added this df conversion bc I thought it was necessary
    line_boxes_df = pd.DataFrame(line_boxes)

    # PNG_NAME = 'output_page1.png'
    # PNG_NAME = png_file_path
    # img = cv2.imread(PNG_NAME)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(png_file_path, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_rgb)

    for idx, row in line_boxes_df.iterrows():
        rect = patches.Rectangle(
            (row["left"], row["top"]),
            row["width"],
            row["height"],
            linewidth=1,
            edgecolor="blue",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.axis("off")
    plt.savefig("data/ocr_output/line_boxes.png")
    plt.show()
    return


def daddy(pdf_path):
    filtered_image_list = filter_pdf(pdf_path)
    pages_with_bounding_boxes = word_boxes(filtered_image_list)
    cleaned_and_sorted = cleaning_data(
        pages_with_bounding_boxes, filtered_image_list[0]
    )
    # more_ cleaning contains sorted data with added columns for line#'s and word#'s
    more_cleaning, line_boxes = word_boxes_to_sentence_boxes(cleaned_and_sorted)
    plot_page(filtered_image_list[0], line_boxes)

    return more_cleaning


# Assign pdf_path, then apply the filters + bounding boxes
pdf_path = "data/raw_pdfs/textbook_pdf_3_includes_diagrams.pdf"

df_ocr = daddy(pdf_path)

"""
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
"""
