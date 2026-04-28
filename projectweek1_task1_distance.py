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

# --- SPEED CALIBRATION ---
MEASURED_PWM = 150
MEASURED_SPEED_CM_S = 47
SPEED_RATIO = MEASURED_PWM / MEASURED_SPEED_CM_S


def speed_to_duty(speed_cm_s):
    pwm = speed_cm_s * SPEED_RATIO
    pwm = min(pwm, 255)           # limit to max PWM
    duty = (pwm / 255) * 100      # convert to duty cycle
    return duty


def forward(speed):
    duty = speed_to_duty(speed)

    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

    pwm_left.ChangeDutyCycle(duty)
    pwm_right.ChangeDutyCycle(duty)


def stop():
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
    pwm_left.ChangeDutyCycle(0)
    pwm_right.ChangeDutyCycle(0)


# --- MAIN PROGRAM ---
try:
    speed = float(input("Enter target speed (cm/s): "))
    distance = float(input("Enter target distance (cm): "))

    if speed <= 0 or distance <= 0:
        raise ValueError

    run_time = distance / speed   # Time = Distance / Speed

    print("\n--- RUN INFO ---")
    print(f"Speed    : {speed:.2f} cm/s")
    print(f"Distance : {distance:.2f} cm")
    print(f"Time     : {run_time:.2f} s")

    forward(speed)
    time.sleep(run_time)
    stop()

    print("\nMovement complete.")

except ValueError:
    print("Invalid input. Speed and distance must be positive numbers.")

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    stop()
    GPIO.cleanup()
