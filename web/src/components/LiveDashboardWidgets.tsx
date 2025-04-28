import React, { useEffect, useState } from "react";
import { Card, Statistic, Row, Col } from "antd";

export default function LiveDashboardWidgets() {
  const [sys, setSys] = useState<any>({});
  useEffect(() => {
    const fetchSys = () => {
      fetch("/api/v1/monitoring/system-info")
        .then(r => r.json())
        .then(setSys);
    };
    fetchSys();
    const timer = setInterval(fetchSys, 5000);
    return () => clearInterval(timer);
  }, []);

  return (
    <Row gutter={16} style={{ marginBottom: 24 }}>
      <Col span={6}>
        <Card>
          <Statistic title="CPU (%)" value={sys.cpu_percent} suffix="%" />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic title="Memory Used (MB)" value={sys.memory ? Math.round(sys.memory.used / 1024 / 1024) : 0} />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic title="Platform" value={sys.platform} />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic title="Hostname" value={sys.hostname} />
        </Card>
      </Col>
    </Row>
  );
}