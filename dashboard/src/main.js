const mqtt = require("mqtt");

// Node.js Client Options (MQTTS/TCP)
var options = {
  host: "8a62b91fd60f40e7b15cc35bebeca3c0.s1.eu.hivemq.cloud",
  port: 8883,
  protocol: "mqtts",
  username: "group-9",
  password: "Group-9-engineering",
};

var client = mqtt.connect(options);

// Simulation State
let temp = 60.0;
let stirSpeed = 500;
let phLevel = 7.0;

client.on("connect", function () {
  console.log("Simulator Connected to HiveMQ Cloud");

  setInterval(() => {
    const tempChange = (Math.random() - 0.5) * 2;
    temp = Math.max(55, Math.min(85, temp + tempChange));
    client.publish("heating", temp.toFixed(2));

    const stirChange = (Math.random() - 0.5) * 50;
    stirSpeed = Math.max(200, Math.min(1000, stirSpeed + stirChange));
    client.publish("stirring", stirSpeed.toFixed(0));

    const phChange = (Math.random() - 0.5) * 0.2;
    phLevel = Math.max(6.0, Math.min(8.0, phLevel + phChange));
    client.publish("ph", phLevel.toFixed(2));

    console.log(
      `Published: Temp=${temp.toFixed(2)}, Stir=${stirSpeed.toFixed(0)}, pH=${phLevel.toFixed(2)}`,
    );
  }, 1000);
});

client.on("error", function (error) {
  console.log("MQTT Error:", error);
});
