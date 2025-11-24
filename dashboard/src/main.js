const { default: mqtt } = require("mqtt");

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

  setInterval(() => {
    const randomData = Math.random() * 100;
    const topic = "heating";
    client.publish(topic, randomData.toString());
    console.log("Published:", randomData, "on topic:", topic);
  }, 2000);
});

client.on("error", function (error) {
  console.log(error);
});

client.on("message", function (topic, message) {
  console.log("Received message:", message.toString(), "on topic:", topic);
});

client.subscribe("heating");
