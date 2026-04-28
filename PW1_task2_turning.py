import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Motor pins
IN1 = 18   # Left Forward
IN2 = 16   # Left Backward
ENA = 22   # Left Speed

IN3 = 11   # Right Forward
IN4 = 13   # Right Backward
ENB = 15   # Right Speed

GPIO.setup([IN1, IN2, ENA, IN3, IN4, ENB], GPIO.OUT)

pwm_left = GPIO.PWM(ENA, 150)
pwm_right = GPIO.PWM(ENB, 150)

pwm_left.start(0)
pwm_right.start(0)

SPEED = 80  # Duty cycle %

# ?? CALIBRATION VALUE
ROTATION_SPEED = 130
# degrees per second (example)

# ---------------- FUNCTIONS ----------------

def stop():
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
    pwm_left.ChangeDutyCycle(0)
    pwm_right.ChangeDutyCycle(0)

def turn_right(duration):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_left.ChangeDutyCycle(SPEED)
    pwm_right.ChangeDutyCycle(SPEED)
    time.sleep(duration)
    stop()

def turn_left(duration):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_left.ChangeDutyCycle(73)
    pwm_right.ChangeDutyCycle(73)
    time.sleep(duration)
    stop()

# ---------------- MAIN ----------------

try:
    direction = input("Turn left or right (l/r): ").lower()
    angle = float(input("Enter angle (degrees): "))

    # Calculate time needed
    duration = angle / ROTATION_SPEED

    print(f"Turning {direction.upper()} {angle} for {duration:.2f} seconds")

    if direction == 'l':
        turn_left(duration)
    elif direction == 'r':
        turn_right(duration)
    else:
        print("Invalid direction")

except KeyboardInterrupt:
    pass

finally:
    stop()
    GPIO.cleanup()
