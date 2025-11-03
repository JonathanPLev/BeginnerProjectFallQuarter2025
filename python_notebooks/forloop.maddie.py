from pathlib import Path
import fitz  # PyMuPDF
import json

# Folder where your PDFs are stored
pdf_folder = Path("PDFforloop")

# Prepare a list for all PDFs’ data
alldata = []

# Loop through each PDF in the folder
for pdf_path in pdf_folder.glob("*.pdf"):
    print(f"Processing {pdf_path.name}...")
    doc = fitz.open(pdf_path)

    # Loop through each page in the PDF
    for page_num, page in enumerate(doc, start=1):
        page_info = {"file": pdf_path.name, "page": page_num, "content": []}
        page_dict = page.get_text("dict")

        # Loop through blocks, lines, and spans
        for block in page_dict["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    page_info["content"].append({
                        "text": span["text"],
                        "bbox": span["bbox"]
                    })

        alldata.append(page_info)

    doc.close()

# Save results to a JSON file



