#!/bin/bash
set -e

echo "=== MCP Server Otomatik Kurulum Başladı ==="

# 1. Python ortamı oluşturuluyor
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate

# 2. Gereksinimler yükleniyor
if [ -f "poetry.lock" ]; then
  echo "Poetry ile kurulum..."
  pip install poetry
  poetry install
else
  echo "pip ile kurulum..."
  pip install --upgrade pip
  pip install -r requirements.txt
fi

# 3. .env dosyası hazırlanıyor
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ".env dosyası oluşturuldu. Lütfen anahtarlarını kontrol edin."
fi

# 4. Tesseract kontrolü (sistem için gerekli)
if ! command -v tesseract &> /dev/null; then
  echo "HATA: Tesseract OCR sistemde kurulu değil!"
  echo "Ubuntu: sudo apt install tesseract-ocr"
  echo "Mac:    brew install tesseract"
  exit 1
fi

# 5. Veritabanı migrasyonu (SQLite)
python -c "from src.models.database import init_db; init_db()"

# 6. Sunucu başlatma komutu
echo ""
echo "Kurulum tamamlandı."
echo "Sunucuyu başlatmak için:"
echo ""
echo "source venv/bin/activate"
echo "uvicorn src.main:app --reload"
echo ""
echo "API:     http://localhost:8000/docs"
echo "Frontend için:"
echo ""
echo "cd web && npm install && npm start"