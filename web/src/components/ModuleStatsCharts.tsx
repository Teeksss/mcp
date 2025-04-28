import React, { useEffect, useState } from "react";
import { fetchTrafficStats } from "../api/mcp";
import { Line, Bar } from "@ant-design/charts";

export default function ModuleStatsCharts() {
  const [data, setData] = useState<any[]>([]);
  useEffect(() => {
    fetchTrafficStats().then(setData);
  }, []);

  // İstek ve hata grafiği
  const reqConfig = {
    data,
    xField: "timestamp",
    yField: "requests",
    seriesField: "module",
    height: 200,
    title: { visible: true, text: "Saatlik İstek Sayısı" },
    legend: { position: "top" },
  };
  const errConfig = {
    data,
    xField: "timestamp",
    yField: "errors",
    seriesField: "module",
    height: 200,
    color: ["#db2828", "#f2711c"],
    title: { visible: true, text: "Saatlik Hata Sayısı" },
    legend: { position: "top" },
  };
  const latConfig = {
    data,
    xField: "timestamp",
    yField: "latency",
    seriesField: "module",
    height: 200,
    color: ["#2185d0", "#21ba45"],
    title: { visible: true, text: "Ortalama Yanıt Süresi (sn)" },
    legend: { position: "top" },
  };

  return (
    <div>
      <h3>Modül Detaylı Trafik Grafikleri</h3>
      <Line {...reqConfig} />
      <Bar {...errConfig} />
      <Line {...latConfig} />
    </div>
  );
}