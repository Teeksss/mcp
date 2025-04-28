import AlarmPanel from "./AlarmPanel";
import AlarmStatsChart from "./AlarmStatsChart";
// ...
<Row gutter={24} style={{ marginTop: 24 }}>
  <Col span={12}><Card title="Alarmlar"><AlarmPanel /></Card></Col>
  <Col span={12}><Card title="Alarm İstatistikleri"><AlarmStatsChart /></Card></Col>
</Row>