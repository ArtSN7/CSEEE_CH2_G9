import "./Home.css";
import box from "@mui/material/box";
import { linechart } from "@mui/x-charts/linechart";

const margin = { right: 24 };
const udata = [4000, 3000, 2000, 2780, 1890, 2390, 3490, 1000, 200, 300];
const pdata = [2400, 1398, 9800, 3908, 4800, 3800, 4300, 5800, 8900];
function home() {
  return (
    <div classname="home">
      <div classname="topnav">
        <a class="active" href="#home">
          home
        </a>
        <a href="#heating">heating</a>
        <a href="#stirring">stirring</a>
        <a href="#temps">temperature</a>
      </div>

      <box sx={{ width: "100%", height: 300 }}>
        <linechart
          series={[{ data: udata, label: "pv" }]}
          xaxis={[{ scaletype: "point", data: pdata }]}
          yaxis={[{ width: 50 }]}
          margin={margin}
        />
      </box>
    </div>
  );
}

export default home;
