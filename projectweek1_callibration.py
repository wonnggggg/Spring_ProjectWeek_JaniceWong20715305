import RPi.GPIO as GPIO
import time

# --- GPIO SETUP ---
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Left Motor
IN1 = 18
IN2 = 16
ENA = 22

# Right Motor
IN3 = 11
IN4 = 13
ENB = 15

GPIO.setup([IN1, IN2, ENA, IN3, IN4, ENB], GPIO.OUT)

# PWM setup
pwm_left = GPIO.PWM(ENA, 150)
pwm_right = GPIO.PWM(ENB, 150)

pwm_left.start(0)
pwm_right.start(0)

RUN_TIME = 3  # seconds


def forward():

    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

    pwm_left.ChangeDutyCycle((150/255)*100)
    pwm_right.ChangeDutyCycle((150/255)*100)
    
def stop():
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
    pwm_left.ChangeDutyCycle(0)
    pwm_right.ChangeDutyCycle(0)
    
try:
    forward()
    time.sleep(RUN_TIME)

    stop()

finally:
    stop()
    GPIO.cleanup()
