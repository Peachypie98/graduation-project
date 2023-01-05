#!/usr/bin/env python

import RPi.GPIO as GPIO
import time
GPIO.setwarnings(False)

#플레이트 (PWM)
ENA = 32
IN1 = 36
IN2 = 38

# set pin numbers to the board's
GPIO.setmode(GPIO.BOARD)
GPIO.setup([ENA,IN1,IN2], GPIO.OUT, initial=GPIO.LOW)
p1 = GPIO.PWM(ENA, 100)

# Backward
GPIO.output(IN2, GPIO.LOW)
GPIO.output(IN1, GPIO.HIGH)
p1.start(25)
time.sleep(3)
p1.stop()


GPIO.cleanup()