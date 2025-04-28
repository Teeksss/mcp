const API_URL = import.meta.env.VITE_MCP_API || "http://localhost:8000";

export async function fetchModules() {
  const res = await fetch(`${API_URL}/api/v1/pipeline/modules`);
  return res.json();
}

export async function createModule(data: any) {
  const res = await fetch(`${API_URL}/api/v1/pipeline/modules`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
}

export async function updateModule(id: string, data: any) {
  const res = await fetch(`${API_URL}/api/v1/pipeline/modules/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
}

export async function deleteModule(id: string) {
  await fetch(`${API_URL}/api/v1/pipeline/modules/${id}`, { method: "DELETE" });
}

export async function fetchHealth() {
  const res = await fetch(`${API_URL}/health`);
  return res.json();
}

export async function fetchTrafficStats() {
  const res = await fetch(`${API_URL}/api/v1/monitoring/traffic`);
  return res.json();
}

export function subscribeLogs(onMessage: (msg: string) => void) {
  const evtSource = new EventSource(`${API_URL}/api/v1/logs/stream`);
  evtSource.onmessage = (e) => onMessage(e.data);
  return evtSource;
}