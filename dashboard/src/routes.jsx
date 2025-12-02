import Home from "./Home";
import Heating from "./Heating";
import Stirring from "./Stirring";
import PH from "./pH";
import Layout from "./Layout";

export const routes = [
  {
    element: <Layout />,
    children: [
      {
        path: "/",
        element: <Home />,
      },
      {
        path: "/heating",
        element: <Heating />,
      },
      {
        path: "/stirring",
        element: <Stirring />,
      },
      {
        path: "/ph",
        element: <PH />,
      },
    ],
  },
];