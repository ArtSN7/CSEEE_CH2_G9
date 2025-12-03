import React from "react";
import { useMqttTopic } from "./mqttHooks";
import { LineChart } from "@mui/x-charts/LineChart";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Typography from "@mui/material/Typography";
import Grid from "@mui/material/Grid";

const KpiCard = ({ title, value, unit, data, color }) => {
  const chartData = data.length > 0 ? data.map((d) => d.value) : [0];
  const xData = data.length > 0 ? data.map((_, i) => i) : [0];

  return (
    <Card
      variant="outlined"
      sx={{ height: "100%", display: "flex", flexDirection: "column" }}
    >
      <CardContent sx={{ flexGrow: 1 }}>
        <Typography color="textSecondary" gutterBottom>
          {title}
        </Typography>
        <Typography variant="h4" component="div">
          {value != null ? value.toFixed(1) : "--"}{" "}
          <span style={{ fontSize: "0.6em" }}>{unit}</span>
        </Typography>
        <Box sx={{ height: 60, mt: 2 }}>
          <LineChart
            series={[
              {
                data: chartData,
                color: color,
                showMark: false,
                curve: "linear",
              },
            ]}
            xAxis={[{ data: xData, scaleType: "point" }]}
            leftAxis={null}
            bottomAxis={null}
            margin={{ left: 0, right: 0, top: 5, bottom: 5 }}
          />
        </Box>
      </CardContent>
    </Card>
  );
};

function Home() {
  const { data: heatData, current: currentHeat } = useMqttTopic("readings/heating");
  const { data: stirData, current: currentStir } = useMqttTopic("readings/stirring");
  const { data: phData, current: currentPh } = useMqttTopic("readings/ph");

  const minLength = Math.min(heatData.length, stirData.length, phData.length);
  const sliceHeat = heatData.slice(-minLength);
  const slicePh = phData.slice(-minLength);

  return (
    <div className="dashboard-page">
      <Typography variant="h4" sx={{ mb: 3, fontWeight: "bold" }}>
        Lab Process Overview
      </Typography>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <KpiCard
            title="Temperature"
            value={currentHeat?.value}
            unit="°C"
            data={heatData.slice(-20)}
            color="#f44336"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <KpiCard
            title="Stirring Speed"
            value={currentStir?.value}
            unit="RPM"
            data={stirData.slice(-20)}
            color="#2196f3"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <KpiCard
            title="pH Level"
            value={currentPh?.value}
            unit="pH"
            data={phData.slice(-20)}
            color="#4caf50"
          />
        </Grid>
      </Grid>

      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Combined Trends (Live)
          </Typography>
          <Box sx={{ height: 350, width: "100%" }}>
            {minLength > 0 ? (
              <LineChart
                xAxis={[
                  {
                    data: sliceHeat.map((d) => d.timestamp),
                    scaleType: "time",
                  },
                ]}
                series={[
                  {
                    data: sliceHeat.map((d) => d.value),
                    label: "Temp (°C)",
                    color: "#f44336",
                    showMark: false,
                  },
                  {
                    data: slicePh.map((d) => d.value),
                    label: "pH",
                    color: "#4caf50",
                    yAxisKey: "phAxis",
                    showMark: false,
                  },
                ]}
                yAxis={[
                  { id: "default" },
                  { id: "phAxis", scaleType: "linear", min: 0, max: 14 },
                ]}
                rightAxis="phAxis"
                margin={{ left: 50, right: 50, top: 20, bottom: 30 }}
              />
            ) : (
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  height: "100%",
                }}
              >
                <Typography color="textSecondary">
                  Waiting for data...
                </Typography>
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>
    </div>
  );
}

export default Home;
