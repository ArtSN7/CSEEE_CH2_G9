import React from "react";
import { Link, useLocation } from "react-router-dom";
import "./Home.css";
import { useColorMode } from "./ThemeContext";
import { useTheme } from "@mui/material/styles";
import IconButton from "@mui/material/IconButton";
import Brightness4Icon from "@mui/icons-material/Brightness4";
import Brightness6Icon from "@mui/icons-material/Brightness7";
import Box from "@mui/material/Box";

const Navbar = () => {
  const location = useLocation();
  const theme = useTheme();
  const { toggleColorMode } = useColorMode();

  const isActive = (path) => (location.pathname === path ? "active" : "");

  return (
    <nav className="topnav">
      <div className="nav-brand">Lab Dashboard</div>
      <div className="nav-links">
        <Link className={isActive("/")} to="/">
          Overview
        </Link>
        <Link className={isActive("/heating")} to="/heating">
          Heating
        </Link>
        <Link className={isActive("/stirring")} to="/stirring">
          Stirring
        </Link>
        <Link className={isActive("/ph")} to="/ph">
          pH
        </Link>
      </div>
      <Box sx={{ marginLeft: "auto" }} color="White">
        <IconButton sx={{ ml: 1 }} onClick={toggleColorMode} color="inherit">
          {theme.palette.mode === "dark" ? (
            <Brightness6Icon />
          ) : (
            <Brightness4Icon />
          )}
        </IconButton>
      </Box>
    </nav>
  );
};

export default Navbar;
