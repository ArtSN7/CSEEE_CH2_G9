#ifndef COMMUNICATION_H
#define COMMUNICATION_H

struct receiver_state
{
  //unsure what should go in here just yet
};

enum PINOUT
{
  // decide which pins the peripherals are connected to, put them in here with a suitable name and their arduino pin numbers
}

class Receiver
{
  public:
    void Receiver(enum PINOUT pinout, enum PROTOCOL protocol) 
      : pinout(pinout), protocol(protocol) {}

    //in case we changed pinout? might not be needed, unsure
    void setPinout(enum PINOUT pinout_s)
    {
      this->pinout = pinout_s;
    }

    //again, these mightnt be needed
    enum PINOUT getPinout()
    {
      return this->pinout;
    }
    enum PROTOCOL getProtocol()
    {
      return this->protocol;
    }

  protected:
    enum PINOUT pinout;
    enum PROTOCOL protocol;
    struct receiver_state state;
}

class SPI_Receiver : public Receiver
{
  public:
  protected:
  private:
}

class I2C_Receiver : public Receiver
{
  public:
  protected:
  private:
}

#endif