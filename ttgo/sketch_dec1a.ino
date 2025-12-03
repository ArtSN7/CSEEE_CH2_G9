#include <WiFi.h>
#include <Wire.h>
#include <PubSubClient.h>
#include "secrets.h"
#include <WiFiClientSecure.h>
#include "esp_eap_client.h"
#include <TFT_eSPI.h>
#include <bits/stdc++.h>

TFT_eSPI tft = TFT_eSPI();


const char* USER = "zcabca1@ucl.ac.uk";

const char* mqtt_broker = "8a62b91fd60f40e7b15cc35bebeca3c0.s1.eu.hivemq.cloud";
const int mqtt_port = 8883;
const char* mqtt_username = "group-9";
const char* mqtt_password = "Group-9-engineering";

//setpoints
float temperature_setpoint = 30.0;
float pH_setpoint = 5.0;
float rpm_setpoint = 800.0;
//sensor data
float current_temp = 0.0;
float current_pH = 0.0;
float current_rpm = 0.0;

//arduino i2c address
#define SDA_PIN 21
#define SCL_PIN 22
const int ARDUINO_ADDRESS = 8;
long previous_time = 0;

WiFiClientSecure wifiClient;
PubSubClient mqttClient(wifiClient);

void drawInfo() {
  tft.fillScreen(TFT_BLACK);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.setTextSize(2);
    tft.drawString("temp:", 20, 95);
    tft.drawString("pH:", 20, 70);
    tft.drawString("rpm:", 20, 45);
    tft.drawString(String(temperature_setpoint), 90, 95);
    tft.drawString(String(pH_setpoint), 90, 70);
    tft.drawString(String(rpm_setpoint), 90, 45);
    tft.drawString(String(current_temp), 180, 95);
    tft.drawString(String(current_pH), 180, 70);
    tft.drawString(String(current_rpm), 180, 45);
    tft.setTextColor(TFT_RED, TFT_BLACK);
    tft.setTextSize(2);
    tft.drawString("SETP", 90, 20);
    tft.drawString("VAL", 180, 20);
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message;
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  if (String(topic) == "setpoints/ph") {
    pH_setpoint = message.toFloat();
    Serial.printf("\nNew pH %f\n", pH_setpoint);
  }
  if (String(topic) == "setpoints/heating") {
    temperature_setpoint = message.toFloat();
    Serial.printf("\nNew temperature %f\n", temperature_setpoint);
  }
  if (String(topic) == "setpoints/stirring") {
    rpm_setpoint = message.toFloat();
    Serial.printf("\nNew rpm %f\n", rpm_setpoint);
  }
  sendSetpointsToArduino();
}

void readFromArduino() {
  Wire.requestFrom(ARDUINO_ADDRESS, 12);

  byte heatingBytes[4];
  byte phBytes[4];
  byte stirringBytes[4];
  if (Wire.available() == 12) {
    for (int i=0; i<4; i++) {
      heatingBytes[i] = Wire.read();
    }
    for (int i=0; i<4; i++) {
      phBytes[i] = Wire.read();
    }
    for (int i=0; i<4; i++) {
      stirringBytes[i] = Wire.read();
    }
    memcpy(&current_temp, heatingBytes, 4);
    memcpy(&current_pH, phBytes, 4);
    memcpy(&current_rpm, stirringBytes, 4);


    Wire.endTransmission();
    Serial.printf("Received current readings temperature:%f, pH:%f, rpm:%f to Arduino.", current_temp, current_pH, current_rpm);
  } 
  else {
    Serial.print("Failed to receive from Arduino.");
    //TEMPORARY TEMPORARY TEMPORARY
    current_temp = (float)(rand()) / (float)(rand());
    current_pH = (float)(rand()) / (float)(rand());
    current_rpm = (float)(rand()) / (float)(rand());
    //TEMPORARY TEMPORARY TEMPORARY
  }
}

void sendSetpointsToArduino() {
  Wire.beginTransmission(ARDUINO_ADDRESS);
  
  byte* tempBytes = (byte*)&temperature_setpoint;
  Wire.write(tempBytes, 4);
  
  byte* phBytes = (byte*)&pH_setpoint;
  Wire.write(phBytes, 4);
  
  byte* rpmBytes = (byte*)&rpm_setpoint;
  Wire.write(rpmBytes, 4);
  
  Wire.endTransmission();
  Serial.printf("Sent setpoints temperature:%f, pH:%f, rpm:%f to Arduino.\n", temperature_setpoint, pH_setpoint, rpm_setpoint);
}


void setupMQTT() {
  mqttClient.setServer(mqtt_broker, mqtt_port);
  mqttClient.setCallback(mqttCallback);
}

void setupTFT() {
  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);
}

void reconnect() {
  Serial.println("Connecting to MQTT Broker...");
  while (!mqttClient.connected()) {
    Serial.println("Reconnecting to MQTT Broker...");
    String clientId = "ESP32Client-";
    clientId += String(random(0xffff), HEX);
    
    if (mqttClient.connect(clientId.c_str(), mqtt_username, mqtt_password)) {
      Serial.println("Connected to MQTT Broker.");

      mqttClient.subscribe("setpoints/heating");
      mqttClient.subscribe("setpoints/ph");
      mqttClient.subscribe("setpoints/stirring");
    } else {
      Serial.print("Failed with state");
      Serial.print(mqttClient.state());
      Serial.println(" trying again in 5 seconds");
      delay(5000);
    }
  }
}

void EstablishWiFi() {
  WiFi.mode(WIFI_STA);
  esp_eap_client_set_identity((uint8_t *)USER, strlen(USER));
  esp_eap_client_set_username((uint8_t *)USER, strlen(USER));
  esp_eap_client_set_password((uint8_t *)PASS, strlen(PASS));
  esp_wifi_sta_enterprise_enable();
  WiFi.begin("eduroam");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected");
}

void setup() {
  Serial.begin(9600);
  Wire.begin(SDA_PIN, SCL_PIN);


  EstablishWiFi();
  wifiClient.setInsecure();
  
  setupTFT();
  setupMQTT();
  delay(1000);
  sendSetpointsToArduino(); //sends initial setpoints
}

void loop() {
  if (!mqttClient.connected()) {
    reconnect();
  }
  mqttClient.loop();

  long now = millis();
  if (now - previous_time > 1000) { // Publish every 1 second
    previous_time = now;

    //Read sensor values from the arduino and publish them to the broker
    readFromArduino();
    publishSensorData(current_temp, current_pH, current_rpm);
    drawInfo();
  }
}

void publishSensorData(float temp, float pH, float rpm) {
  mqttClient.publish(("readings/heating"), String(temp, 2).c_str());
  mqttClient.publish(("readings/ph"), String(pH, 2).c_str());
  mqttClient.publish(("readings/stirring"), String(rpm, 2).c_str());
}


