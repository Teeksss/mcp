import React, { useEffect, useState } from "react";
import { Card } from "antd";
import { fetchTrafficStats } from "../api/mcp";
import { Line } from "@ant-design/charts";

export default function ModuleTrafficChart() {
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    fetchTrafficStats().then(setData);
  }, []);

  const config = {
    data,
    xField: "timestamp",
    yField: "count",
    seriesField: "module",
    height: 200,
    smooth: true,
    legend: { position: "top" },
  };

  return (
    <Card>
      <Line {...config} />
    </Card>
  );
}