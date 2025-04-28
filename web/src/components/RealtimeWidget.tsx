import React, { useEffect, useState } from "react";
import { Card, Statistic } from "antd";

export default function RealtimeWidget() {
  const [stats, setStats] = useState({ latency: 0, error_rate: 0, qps: 0 });

  useEffect(() => {
    const interval = setInterval(() => {
      fetch("/api/v1/monitoring/realtime")
        .then(r => r.json())
        .then(setStats);
    }, 2000); // 2 sn'de bir güncelle
    return () => clearInterval(interval);
  }, []);

  return (
    <Card title="Canlı Sistem Durumu" style={{ width: 300 }}>
      <Statistic title="Latency (sn)" value={stats.latency} precision={2} />
      <Statistic title="Error Rate (%)" value={stats.error_rate * 100} precision={2} />
      <Statistic title="QPS" value={stats.qps} />
    </Card>
  );
}