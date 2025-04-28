import React, { useEffect, useState } from "react";
import { fetchHealth } from "../api/mcp";
import { Badge } from "antd";

export default function HealthOverview() {
  const [status, setStatus] = useState<string>("...");
  useEffect(() => {
    fetchHealth().then((d) => setStatus(d.status));
  }, []);
  return (
    <div>
      <Badge
        status={status === "ok" ? "success" : "error"}
        text={status === "ok" ? "Sistem Sağlıklı" : "Sorun Var"}
      />
    </div>
  );
}