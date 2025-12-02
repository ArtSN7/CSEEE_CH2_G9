import React from "react";
import { useMqttTopic } from "./mqttHooks";
import { LineChart } from "@mui/x-charts/LineChart";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Typography from "@mui/material/Typography";
import SetpointInput from "./SetpointInput";

function Heating() {
  const { data, current } = useMqttTopic("heating");

  return (
    <div className="dashboard-page">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#d32f2f' }}>
          Temperature Monitor
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <SetpointInput label="Heating" unit="°C" color="#d32f2f" />
          <Card variant="outlined" sx={{ minWidth: 200 }}>
            <CardContent sx={{ py: 1, '&:last-child': { pb: 1 } }}>
              <Typography variant="subtitle2" color="textSecondary">Current Temp</Typography>
              <Typography variant="h5" color="#d32f2f">
                {current ? `${current.value.toFixed(2)} °C` : "--"}
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
                label: "Temperature (°C)",
                color: "#d32f2f",
                showMark: false,
              },
            ]}
            yAxis={[{ label: 'Temperature (°C)' }]}
            margin={{ left: 70, right: 20, top: 20, bottom: 30 }}
            grid={{ vertical: true, horizontal: true }}
          />
        </Box>
      </Card>
    </div>
  );
}

export default Heating;