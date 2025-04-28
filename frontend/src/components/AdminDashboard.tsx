import React from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useQuery } from '@tanstack/react-query';
import { AdminService } from '../services/AdminService';

export const AdminDashboard: React.FC = () => {
  const { data: metrics, isLoading: metricsLoading } = useQuery(
    ['system-metrics'],
    AdminService.getSystemMetrics,
    {
      refetchInterval: 30000, // Refresh every 30 seconds
    }
  );

  const { data: activeUsers } = useQuery(
    ['active-users'],
    AdminService.getActiveUsers
  );

  const { data: modelStats } = useQuery(
    ['model-stats'],
    AdminService.getModelStats
  );

  if (metricsLoading) {
    return <CircularProgress />;
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Grid container spacing={3}>
        {/* System Health */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              System Health
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={metrics?.systemHealth}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="cpu"
                  stroke="#8884d8"
                  name="CPU Usage"
                />
                <Line
                  type="monotone"
                  dataKey="memory"
                  stroke="#82ca9d"
                  name="Memory Usage"
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Model Performance */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Model Performance
            </Typography>
            <List>
              {modelStats?.map((stat) => (
                <ListItem key={stat.modelId}>
                  <ListItemText
                    primary={stat.modelName}
                    secondary={`Avg. Response Time: ${stat.avgResponseTime}ms | Success Rate: ${stat.successRate}%`}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        {/* Active Users */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Active Users
            </Typography>
            <Typography variant="h3" align="center">
              {activeUsers?.count || 0}
            </Typography>
            <List>
              {activeUsers?.recent.map((user) => (
                <ListItem key={user.id}>
                  <ListItemText
                    primary={user.email}
                    secondary={`Last Active: ${new Date(
                      user.lastActive
                    ).toLocaleString()}`}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        {/* System Alerts */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              System Alerts
            </Typography>
            <List>
              {metrics?.alerts.map((alert) => (
                <ListItem key={alert.id}>
                  <ListItemText
                    primary={alert.message}
                    secondary={alert.timestamp}
                    sx={{
                      color:
                        alert.severity === 'high'
                          ? 'error.main'
                          : 'text.primary',
                    }}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};