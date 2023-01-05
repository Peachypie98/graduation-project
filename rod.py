#!/usr/bin/env python

import RPi.GPIO as GPIO
import time
GPIO.setwarnings(False)

# 막대 (PWM)
ENA = 33
IN1 = 35
IN2 = 37

# set pin numbers to the board's
GPIO.setmode(GPIO.BOARD)
GPIO.setup([ENA,IN1,IN2], GPIO.OUT, initial=GPIO.LOW)
p1 = GPIO.PWM(ENA, 100)

#Going up
# GPIO.output(IN2, GPIO.LOW)
# GPIO.output(IN1, GPIO.HIGH)
# p1.start(40)
# time.sleep(10)
# p1.stop()

#Going down
GPIO.output(IN2, GPIO.HIGH)
GPIO.output(IN1, GPIO.LOW)
p1.start(40)
time.sleep(10.5)
p1.stop()

GPIO.cleanup()