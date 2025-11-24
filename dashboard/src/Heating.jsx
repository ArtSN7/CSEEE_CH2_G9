import Box from "@mui/material/Box";
import { LineChart } from "@mui/x-charts/LineChart";
const { default: mqtt } = require("mqtt");

const margin = { right: 24 };
function Heating() {
  var temps = [4000, 3000, 2000, 2780, 1890, 2390, 3490, 1000, 200, 300];
  var time = [2400, 1398, 9800, 3908, 4800, 3800, 4300, 5800, 8900];
  var options = {
    host: "8a62b91fd60f40e7b15cc35bebeca3c0.s1.eu.hivemq.cloud",
    port: 8883,
    protocol: "mqtts",
    username: "group-9",
    password: "Group-9-engineering",
  };

  var client = mqtt.connect(options);

  client.on("connect", function () {
    console.log("Connected");
  });

  client.on("error", function (error) {
    console.log(error);
  });

  client.on("message", function (topic, message) {
    temps.push(parseFloat(message.toString().toInt));
    time.push();
  });

  client.subscribe("heating");
  return (
    <div>
      <h1 style={{ textAlign: "center" }}>Heating</h1>
      <Box
        sx={{
          width: "100%",
          height: 300,
          display: "flex",
          justifyContent: "center",
        }}
      >
        <LineChart
          series={[{ data: temps, label: "pv" }]}
          xAxis={[{ scaleType: "point", data: time }]}
          yAxis={[{ width: 50 }]}
          margin={margin}
        />
      </Box>
    </div>
  );
}

export default Heating;
