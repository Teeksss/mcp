# GitHub Üzerinden MCP Server Kurulumu

Aşağıdaki adımları izleyerek projeyi GitHub üzerinden hızlıca kurabilirsiniz.

---

## 1. Repoyu Klonla

```bash
git clone https://github.com/KULLANICI_ADINIZ/REPO_ADI.git
cd REPO_ADI
```

---

## 2. Otomatik Kurulum

Hazır kurulum için (Linux/Mac):

```bash
chmod +x setup.sh
./setup.sh
```

Kurulum otomatik tamamlanacak, `.env` dosyanızı kontrol edin ve gerekiyorsa anahtarları doldurun.

---

## 3. Sunucuyu Başlat

```bash
source venv/bin/activate
uvicorn src.main:app --reload
```
- API arayüzü: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 4. Frontend (Web) Arayüzü

```bash
cd web
npm install
npm start
```
- Tarayıcıdan: [http://localhost:3000](http://localhost:3000)

---

## 5. Notlar

- **Tesseract** sisteminizde yüklü olmalı (OCR için).
- Python 3.9+ ve Node.js 16+ önerilir.
- Açılan `.env` dosyasına OpenAI API anahtarı ve diğer gerekli bilgileri giriniz.

---