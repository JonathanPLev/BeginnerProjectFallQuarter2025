# OCR Project Summary (Custom Line-Level OCR)

## Overview
You’re building an OCR (Optical Character Recognition) system that converts scanned PDF pages into digital text.  
Since many PDFs contain only images of text, your pipeline extracts each text line as an image and trains a model to read the text within it.

---

## Project Flow

### 1. Input
- Start with scanned or image-only PDFs.  
- Use `pdf2image` to render each page as an image (`.png` or `.jpg`).

### 2. Preprocessing
- Convert the page to grayscale for simplicity.  
- Apply thresholding (e.g., Otsu) to separate text from background.  
- Use morphological operations and `cv2.findContours()` to detect bounding boxes of each text line.  
- Filter out boxes that are too large or irregular (to ignore diagrams or images).  
- Crop each detected line into its own small image.

Result: a dataset of labeled line images and corresponding text for training.

---

## Model Architecture — CRNN

Your model is a Convolutional Recurrent Neural Network (CRNN), combining CNNs for visual features and RNNs for sequence modeling.

### Structure
```
Input line image (H×W)
   ↓
CNN → extracts visual features per column
   ↓
Flatten along width → sequence of feature vectors
   ↓
BiLSTM → models character sequence (left→right)
   ↓
Linear layer → predicts character probabilities at each timestep
   ↓
CTC loss → aligns predicted sequence with ground truth text
```

### Key Ideas
- CNN handles visual feature extraction.  
- LSTM learns ordering of characters.  
- CTC (Connectionist Temporal Classification) aligns variable-length predictions with target strings, so you don’t need per-character labels.

---

## Training the Model

1. **Prepare dataset**
   - Each sample: `(image, text)`
   - Resize all images to fixed height (e.g., 32 px) and normalize pixels.
   - Convert text to indices using a character-to-ID mapping.

2. **Training loop**
   - Forward pass: images → predicted character sequences.  
   - Compute CTC loss between predictions and target sequences.  
   - Backpropagate and optimize with Adam or SGD.

3. **Monitor performance**
   - Track loss and character-level accuracy.  
   - Optionally visualize decoded predictions versus labels.

---

## Inference (Prediction)
- Preprocess a new text-line image (grayscale, resize).  
- Run it through the trained CRNN.  
- Apply CTC decoding to collapse repeated characters and remove blanks.  
- Output the recognized text string.

If you repeat this for all detected lines in a page, then combine them top-to-bottom, you get the full textual version of your PDF.

---

## Tools & Libraries

| Purpose | Library |
|----------|----------|
| PDF → Image conversion | `pdf2image` |
| Image preprocessing & line detection | `opencv-python` |
| Model & training | `torch`, `torchvision` |
| Visualization & debugging | `matplotlib`, `cv2` |

---

## End Result
- You can feed in scanned PDFs or image pages.  
- Your model outputs plain text lines.  
- You can later combine those into a text file or searchable PDF layer.
