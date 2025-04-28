import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

def pdf_to_texts(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for page in doc:
        text = page.get_text()
        if not text.strip():
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            text = pytesseract.image_to_string(img, lang="eng+tur")
        texts.append(text)
    return texts

async def ocr_image(img_bytes, lang="eng+tur"):
    img = Image.open(img_bytes)
    return pytesseract.image_to_string(img, lang=lang)