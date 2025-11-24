import React from "react";
import "./Home.css";
import { Link } from "react-router-dom";
import { useLocation } from "react-router-dom";

const Navbar = () => {
  const location = useLocation();
  return (
    <div className="topnav">
      <Link className={location.pathname === "/" ? "active" : ""} to="/">
        Home
      </Link>
      <Link
        className={location.pathname === "/heating" ? "active" : ""}
        to="/heating"
      >
        Heating
      </Link>
      <Link
        className={location.pathname === "/stirring" ? "active" : ""}
        to="/stirring"
      >
        Stirring
      </Link>
      <Link
        className={location.pathname === "/temperature" ? "active" : ""}
        to="/temperature"
      >
        Temperature
      </Link>
    </div>
  );
};

export default Navbar;
