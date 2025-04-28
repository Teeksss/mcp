import React, { useEffect } from "react";
import { notification } from "antd";

export default function LiveAlarmToast() {
  useEffect(() => {
    const evt = new EventSource("/api/v1/alarms/stream");
    evt.onmessage = (e) => {
      const alarm = JSON.parse(e.data);
      notification[alarm.level === "critical" ? "error" : alarm.level === "warning" ? "warning" : "info"]({
        message: "Yeni Alarm",
        description: alarm.message,
        duration: 4,
      });
    };
    return () => evt.close();
  }, []);
  return null;
}