# MCP Server: Akıllı Multi-Model + RAG Destekli Sunucu Mimarisi

## 1. Ana Bileşenler

### 1.1 Inference Pipeline (`src/services/pipeline/inference_pipeline.py`)
- Sorguları önbellekten (cache) kontrol eder.
- Model sürümünü belirler ve yük dengeleyiciyle uygun node'u seçer.
- Performans analizine dayalı optimizasyon uygular.
- Model inference'ı gerçekleştirir, hata durumunda error recovery mekanizması devreye girer.
- Sonuçları cache'e yazar ve telemetri/metrik kaydeder.

### 1.2 Performance Analyzer (`src/services/optimization/performance_analyzer.py`)
- Model inference geçmişini ve performans metriklerini izler.
- Latency, memory, throughput gibi metrikleri analiz eder.
- Optimizasyon gereksinimlerini otomatik belirler ve öneriler sunar.

### 1.3 Performance Optimizer (`src/services/optimization/performance_optimizer.py`)
- Komponent bazında darboğaz tespiti ve öneriler üretir.
- Performans geçmişini günceller.

### 1.4 RAG Enhancer (`src/services/intelligence/rag_enhancer.py`)
- Sorgular için dış bağlam (context) getirir ve reranking yapar.
- Sorguları bağlamla zenginleştirir ve cache yönetimi sağlar.

### 1.5 Model ve Deney Yönetimi (`src/models/database.py`)
- SQLAlchemy tabanlı model ve deney ilişkileri.
- Deneylerin durumu, sonuçları ve geçmişi tutulur.

### 1.6 Model Manager (`src/services/intelligence/model_manager.py`)
- Modellerin profili, batch boyutları, desteklediği görevler ve fallback modelleri yönetir.

## 2. Entegrasyon Akışı

1. **Kullanıcıdan sorgu alınır.**
2. Inference pipeline önbelleği kontrol eder.
3. Model sürümü ve node seçimi yapılır (multi-model, load balancing).
4. Gerekirse pipeline, RAG Enhancer ile ek bağlam toplar ve sorguyu iyileştirir (RAG destekli akış).
5. Optimizasyon ve performans kontrolleri uygulanır.
6. Model inference çalıştırılır, hata olursa error recovery çalışır.
7. Sonuç cache'e yazılır, telemetri/metrik kaydı yapılır.
8. Performance analyzer ve optimizer, modelin performansını izler ve önerilerde bulunur.
9. Model ve deneyler veritabanında kayıt altına alınır.

## 3. Teknik Özellikler
- Modern Python (async/await), Prometheus metrikleri, ayrık ve ölçeklenebilir modüller.
- Otomatik hata yönetimi, gelişmiş loglama ve telemetri.
- API tabanlı entegrasyon ve kolay test edilebilirlik.
- Çoklu model ve çoklu görev desteği.

---

**Bu mimari, projenin kod yapısına ve fonksiyonlarına göre entegre edilmiştir. Her bir modül, bağımsız olarak gelişmiş olsa da, birlikte Akıllı Multi-Model + RAG Destekli MCP Server altyapısını bütüncül şekilde oluşturur.**

---

> Daha detaylı kod seviyesinde entegrasyon diyagramı veya örnek kullanım akışı isterseniz belirtin!