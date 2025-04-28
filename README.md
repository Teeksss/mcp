# MCP Server

Kurumsal düzeyde, çoklu model ve RAG destekli Model Kontrol Platformu (MCP) sunucusu.

## Genel Özellikler

- Çoklu Model Yönetimi (versioning, registry)
- Akıllı Yönlendirme & RAG Pipeline
- Sağlam API ve servis mimarisi
- Otomatik test, izleme, logging
- Geliştirici dostu yapı

## Hızlı Başlangıç

```bash
git clone https://github.com/Teeksss/mcp.git
cd mcp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run_server.py
```

## API Dökümantasyonu

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Katkı ve Test

Her katkı PR'ı otomatik testten geçmelidir.

```bash
pytest
flake8 src/
```

Daha fazla detay için [CONTRIBUTING.md](CONTRIBUTING.md) dosyasına bakın.