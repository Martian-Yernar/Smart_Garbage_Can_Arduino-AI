#include <Servo.h>

Servo top;
Servo bottom;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  top.attach(10);
  bottom.attach(9);
  
  bottom.write(0);
  top.write(90);
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0){
    int msg = Serial.parseInt();

    bottom.write(msg);
    delay(1000);
    top.write(0);
    delay(1000); 
    top.write(90);
    
  }
}

