import React, { useEffect, useState } from "react";
import { Alert, List } from "antd";

export default function AlarmPanel() {
  const [alarms, setAlarms] = useState([]);
  useEffect(() => {
    fetch("/api/v1/monitoring/alarms").then(r => r.json()).then(setAlarms);
  }, []);
  return (
    <List
      dataSource={alarms}
      renderItem={alarm => (
        <List.Item>
          <Alert message={alarm.msg} type={alarm.level} showIcon />
        </List.Item>
      )}
    />
  );
}