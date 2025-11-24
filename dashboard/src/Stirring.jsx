import Box from "@mui/material/Box";
import { LineChart } from "@mui/x-charts/LineChart";
const margin = { right: 24 };
const udata = [4000, 3000, 2000, 2780, 1890, 2390, 3490, 1000, 200, 300];
const pdata = [2400, 1398, 9800, 3908, 4800, 3800, 4300, 5800, 8900];
function Stirring() {
  return (
    <div>
      <h1 style={{ textAlign: "center" }}>Stirring</h1>
      <Box sx={{ width: "100%", height: 300, display: "flex", justifyContent: "center" }}>
        <LineChart
          series={[{ data: udata, label: "pv" }]}
          xAxis={[{ scaleType: "point", data: pdata }]}
          yAxis={[{ width: 50 }]}
          margin={margin}
        />
      </Box>
    </div>
  );
}

export default Stirring;
