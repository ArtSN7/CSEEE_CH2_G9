import React from "react";
import { useMqttTopic } from "./mqttHooks";
import { LineChart } from "@mui/x-charts/LineChart";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Typography from "@mui/material/Typography";
import SetpointInput from "./SetpointInput";

function Stirring() {
  const { data, current } = useMqttTopic("stirring");

  return (
    <div className="dashboard-page">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#1976d2' }}>
          Stirring Speed
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <SetpointInput label="Stirring" unit="RPM" color="#1976d2" />
          <Card variant="outlined" sx={{ minWidth: 200 }}>
            <CardContent sx={{ py: 1, '&:last-child': { pb: 1 } }}>
              <Typography variant="subtitle2" color="textSecondary">Current Speed</Typography>
              <Typography variant="h5" color="#1976d2">
                {current ? `${Math.round(current.value)} RPM` : "--"}
              </Typography>
            </CardContent>
          </Card>
        </Box>
      </Box>

      <Card variant="outlined" sx={{ p: 2 }}>
        <Box sx={{ height: 500, width: '100%' }}>
          <LineChart
            xAxis={[{ 
              data: data.map((d) => d.timestamp), 
              scaleType: 'time', 
              valueFormatter: (date) => date.toLocaleTimeString(),
            }]}
            series={[
              {
                data: data.map((d) => d.value),
                label: "Speed (RPM)",
                color: "#1976d2",
                showMark: false,
              },
            ]}
            yAxis={[{ label: 'RPM' }]}
            margin={{ left: 70, right: 20, top: 20, bottom: 30 }}
            grid={{ vertical: true, horizontal: true }}
          />
        </Box>
      </Card>
    </div>
  );
}

export default Stirring;