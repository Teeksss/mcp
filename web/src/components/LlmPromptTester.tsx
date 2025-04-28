import React, { useState } from "react";
import { Select, Input, Button, Typography, Card } from "antd";

const { TextArea } = Input;
const { Title } = Typography;

const LLM_OPTIONS = [
  { value: "phi3", label: "Phi-3" },
  { value: "gpt4", label: "GPT-4" }
];

export default function LlmPromptTester() {
  const [model, setModel] = useState("phi3");
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const runPrompt = async () => {
    setLoading(true);
    setResult("");
    const resp = await fetch("/api/v1/llm/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_key: model, prompt, max_tokens: 128 })
    });
    const data = await resp.json();
    setResult(data.result || "");
    setLoading(false);
  };

  return (
    <Card style={{ maxWidth: 600, margin: "auto", marginTop: 32 }}>
      <Title level={4}>LLM Prompt Test</Title>
      <Select options={LLM_OPTIONS} value={model} onChange={setModel} style={{ width: 140, marginBottom: 16 }} />
      <TextArea
        rows={4}
        placeholder="Prompt girin..."
        value={prompt}
        onChange={e => setPrompt(e.target.value)}
        style={{ marginBottom: 16 }}
      />
      <Button type="primary" onClick={runPrompt} loading={loading}>Çalıştır</Button>
      {result && (
        <Card style={{ marginTop: 24, background: "#fafafa" }}>
          <b>LLM Yanıtı:</b>
          <div style={{ whiteSpace: "pre-wrap" }}>{result}</div>
        </Card>
      )}
    </Card>
  );
}