#ifndef SENSOR_H
#define SENSOR_H

struct sensor_reading
{
  double value;
  uint32_t timestamp;
  bool valid;
};

class Sensor
{
  public:
    Sensor()
    SPI_Receiver* getReceiver();
    struct sensor_reading getReading();
  protected:
    SPI_Receiver receiver;
    struct sensor_reading reading;
  private:
}

//these might be unneeded, but better for clarity, most is abstracted already in Sensor class
class PH_Sensor : public Sensor
{
  public:
  protected:
  private:
}

class RPM_Sensor : public Sensor
{
  public:
  protected:
  private:
}

class Temperature_Sensor : public Sensor
{
  public:
  protected:
  private:
}

#endif
