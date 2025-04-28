import io
from src.services.rag.pdf_ocr_utils import pdf_to_texts, ocr_image

async def process_file(file, ocr_lang="eng+tur"):
    filename = file.filename.lower()
    content = await file.read()
    docs = []
    if filename.endswith(".pdf"):
        for page_idx, text in enumerate(pdf_to_texts(io.BytesIO(content))):
            docs.append({"content": text, "metadata": {"filename": file.filename, "page": page_idx+1}})
    elif any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]):
        text = await ocr_image(io.BytesIO(content), ocr_lang)
        docs.append({"content": text, "metadata": {"filename": file.filename}})
    else:
        docs.append({"content": content.decode(), "metadata": {"filename": file.filename}})
    return docs

def fetch_web_content(url):
    import requests
    from bs4 import BeautifulSoup
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    for s in soup(["script", "style"]): s.decompose()
    return soup.get_text(separator="\n", strip=True)