# LLM Entegrasyon Yapısı

## Özellikler
- Hem yerel (transformers tabanlı) hem bulut (OpenAI, vs.) LLM entegrasyonu
- ModelManager ile çoklu model kaydı, seçimi ve çağrısı
- API üzerinden kolay prompt gönderme ve LLM cevabı alma
- Kolay genişletilebilirlik (Anthropic, VertexAI, Replicate, vb.)

## Örnek Kullanım

```python
from src.services.intelligence.model_manager import ModelManager

mm = ModelManager()
mm.register("llama", "transformers", "meta-llama/Llama-2-7b-chat-hf", device="cuda")
mm.register("gpt4", "openai", "gpt-4", api_key="sk-...")

answer = mm.generate("llama", "Türkiye'nin başkenti nedir?")
```

## API Örneği

`POST /api/v1/llm/generate`

```json
{
  "model_key": "gpt4",
  "prompt": "What is the capital of Turkey?",
  "max_tokens": 64,
  "temperature": 0.2
}
```

---

Ek sağlayıcılar için (Anthropic, VertexAI, Replicate, vb.), `LLMClient`'a yeni bir "provider" bloğu eklemeniz yeterlidir.