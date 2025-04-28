# MCP Server (Multi-Model + RAG + LLM Platform)

## Kurulum Rehberi

### Gereksinimler

- **Python 3.9+**
- **Node.js 16+** (frontend için)
- **Tesseract** (OCR desteği için)
- (Linux/Mac: `sudo apt install tesseract-ocr` veya `brew install tesseract`)
- **pip** veya **poetry** (isteğe bağlı)

---

## 1. Backend Kurulumu

### a) Sanal Ortam Oluştur

```bash
python -m venv venv
source venv/bin/activate
```

### b) Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
# veya
poetry install
```

### c) Ortam Değişkenleri

`.env` dosyasını oluştur:

```bash
cp .env.example .env
```
Gerekirse `OPENAI_API_KEY` ve diğer alanları doldur.

### d) Veritabanını Başlat

```bash
python -c "from src.models.database import init_db; init_db()"
```

### e) Sunucuyu Çalıştır

```bash
uvicorn src.main:app --reload
```
- Uygulama arayüzü: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 2. Frontend (Web) Kurulumu

```bash
cd web
npm install
npm start
```
- Arayüz: [http://localhost:3000](http://localhost:3000)

---

## 3. Notlar

- PDF/OCR için Tesseract kurulmalı.
- LLM entegrasyonu için `OPENAI_API_KEY` veya HuggingFace modeli indirecek internet bağlantısı gereklidir.
- Vektör veritabanı ve LLM eklemek için ilgili Python dosyalarından kolayca genişletebilirsiniz.

---