import React, { useState } from "react";
import { publishMqtt } from "./mqttHooks";
import Box from "@mui/material/Box";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";

function SetpointInput({ label, unit, color }) {
  const [value, setValue] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (value.trim() !== "") {
      const message = value;
      publishMqtt(`setpoints/${label.toLowerCase()}`, message);
      setValue("");
    }
  };

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
      <TextField
        size="small"
        type="number"
        label={`Set ${label}`}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        inputProps={{ step: "any" }}
        sx={{ width: 150 }}
      />
      <Typography variant="body2" sx={{ minWidth: 40 }}>{unit}</Typography>
      <Button
        type="submit"
        variant="contained"
        size="small"
        sx={{ bgcolor: color, '&:hover': { bgcolor: color, opacity: 0.9 } }}
      >
        Send
      </Button>
    </Box>
  );
}

export default SetpointInput;
