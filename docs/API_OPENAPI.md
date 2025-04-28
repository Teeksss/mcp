# OpenAPI/Swagger Şeması

API dökümantasyonuna otomatik olarak `/docs` ve `/redoc` endpointlerinden erişebilirsiniz.

Elle OpenAPI şeması örneği:
```json
{
  "openapi": "3.0.2",
  "info": {
    "title": "MCP Server - Akıllı Multi-Model + RAG Destekli",
    "version": "0.2.0"
  },
  "paths": {
    "/api/v1/pipeline/execute": {
      "post": {
        "summary": "Execute the inference pipeline for the given query and configuration.",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/PipelineRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/PipelineResponse"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "PipelineConfig": {
        "title": "PipelineConfig",
        "type": "object",
        "properties": {
          "model_type": { "type": "string" },
          "max_tokens": { "type": "integer", "default": 100 },
          "temperature": { "type": "number", "default": 0.7 }
        },
        "required": ["model_type"]
      },
      "PipelineRequest": {
        "title": "PipelineRequest",
        "type": "object",
        "properties": {
          "query": { "type": "string" },
          "config": { "$ref": "#/components/schemas/PipelineConfig" }
        },
        "required": ["query", "config"]
      },
      "PipelineResponse": {
        "title": "PipelineResponse",
        "type": "object",
        "properties": {
          "result": { "type": "object" },
          "processing_time": { "type": "number" },
          "status": { "type": "string", "default": "success" },
          "error": { "type": "string", "nullable": True }
        }
      }
    }
  }
}
```