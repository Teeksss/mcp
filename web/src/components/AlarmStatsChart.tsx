import React, { useEffect, useState } from "react";
import { Pie } from "@ant-design/charts";

export default function AlarmStatsChart() {
  const [alarms, setAlarms] = useState<any[]>([]);
  useEffect(() => {
    fetch("/api/v1/alarms").then(r => r.json()).then(setAlarms);
  }, []);
  const data = [
    { type: "Kritik", value: alarms.filter(a => a.level === "critical").length },
    { type: "Uyarı", value: alarms.filter(a => a.level === "warning").length },
    { type: "Bilgi", value: alarms.filter(a => a.level === "info").length },
  ];
  return <Pie data={data} angleField="value" colorField="type" />;
}