import { useState, useEffect } from "react";
import mqtt from "mqtt";

const MQTT_OPTIONS = {
  protocol: "wss",
  host: "8a62b91fd60f40e7b15cc35bebeca3c0.s1.eu.hivemq.cloud",
  port: 8884,
  path: "/mqtt",
  username: "group-9",
  password: "Group-9-engineering",
  clientId: "lab_dashboard_group9",
  keepalive: 60,
  reconnectPeriod: 1000,
  clean: true,
};

let client = null;
const topicListeners = new Map();
function getMqttClient() {
  if (!client) {
    console.log("Initializing", {
      ...MQTT_OPTIONS,
    });
    client = mqtt.connect(MQTT_OPTIONS);

    client.on("connect", () => {
      console.log("MQTT Connected via WSS");
      topicListeners.forEach((_, topic) => {
        client.subscribe(topic, (err) => {
          if (err) console.error(`Failed to subscribe to ${topic}`, err);
          else console.log(`Subscribed to ${topic}`);
        });
      });
    });

    client.on("error", (err) => {
      console.error("MQTT Connection Error:", err);
    });

    client.on("offline", () => {
      console.log("MQTT Client Offline");
    });

    client.on("message", (topic, message) => {
      console.log(`MQTT Message on ${topic}: ${message.toString()}`);
      if (topicListeners.has(topic)) {
        const value = parseFloat(message.toString());
        if (!isNaN(value)) {
          const dataPoint = {
            value,
            timestamp: new Date(),
          };
          topicListeners.get(topic).forEach((callback) => callback(dataPoint));
        }
      }
    });
  }
  return client;
}

export const useMqttTopic = (topic, bufferSize = 100) => {
  const [data, setData] = useState([]);
  const [current, setCurrent] = useState(null);

  useEffect(() => {
    const mqttClient = getMqttClient();

    if (!topicListeners.has(topic)) {
      topicListeners.set(topic, new Set());
      mqttClient.subscribe(topic);
    }

    const handleMessage = (newDataPoint) => {
      setCurrent(newDataPoint);
      setData((prevData) => {
        const updated = [...prevData, newDataPoint];
        if (updated.length > bufferSize) {
          return updated.slice(updated.length - bufferSize);
        }
        return updated;
      });
    };

    topicListeners.get(topic).add(handleMessage);

    return () => {
      const listeners = topicListeners.get(topic);
      if (listeners) {
        listeners.delete(handleMessage);
      }
    };
  }, [topic, bufferSize]);

  return { data, current };
};

export const publishMqtt = (topic, message) => {
  const mqttClient = getMqttClient();
  mqttClient.publish(topic, message, (err) => {
    if (err) console.error(`Failed to publish to ${topic}`, err);
    else console.log(`Published to ${topic}: ${message}`);
  });
};
