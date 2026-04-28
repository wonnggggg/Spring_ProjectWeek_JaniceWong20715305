import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import time

# --- 1. PIN & SPEED CONFIGURATION ---
ENA, IN1, IN2 = 15, 11, 13
ENB, IN3, IN4 = 22, 18, 16

# Speed Settings
BASE_SPEED = 45      
MAX_SPEED = 90       # Slightly lower than 100 for better control
KP = 0.5             # Proportional Gain (Reacts to current error)
KI = 0.02            # Integral Gain (Corrects long-term offset)

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Setup Pins
motor_pins = [ENA, IN1, IN2, ENB, IN3, IN4]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)

# Setup PWM
pwm_left = GPIO.PWM(ENA, 1000)
pwm_right = GPIO.PWM(ENB, 1000)
pwm_left.start(0)
pwm_right.start(0)

# --- PI VARIABLES ---
integral = 0
last_time = time.time()

def set_motor_speed(left, right):
    """
    Controls motor speed and direction.
    Logic is inverted (LOW/HIGH) to ensure forward movement based on your wiring.
    """
    # --- LEFT MOTOR ---
    if left >= 0: 
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
    else:         
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
    pwm_left.ChangeDutyCycle(abs(left))

    # --- RIGHT MOTOR ---
    if right >= 0: 
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
    else:          
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
    pwm_right.ChangeDutyCycle(abs(right))

def stop_motors():
    set_motor_speed(0, 0)

# --- 2. CAMERA SETUP ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240), "format": "RGB888"})
picam2.configure(config)
picam2.start()

print("PI Line Follower Active! Press 'q' to stop.")

try:
    while True:
        # 1. Capture Frame & Timing
        frame = picam2.capture_array()
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # 2. Crop to Region of Interest (Bottom 80 pixels)
        height, width, _ = frame.shape
        roi = frame[height-80:height, :]
        
        # 3. Image Processing
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        
        # 4. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            if M['m00'] > 500: # Ignore tiny specks of noise
                cx = int(M['m10'] / M['m00'])
                
                # --- 5. PI CONTROL LOGIC ---
                error = cx - (width // 2)
                
                # Accumulate Integral
                integral += error * dt
                
                # Anti-Windup: Limit the integral's influence
                integral = max(min(integral, 500), -500)
                
                # Calculate Correction
                correction = int((error * KP) + (integral * KI))
                
                left_speed = BASE_SPEED + correction
                right_speed = BASE_SPEED - correction
                
                # Constrain speeds
                left_speed = max(min(left_speed, MAX_SPEED), -MAX_SPEED)
                right_speed = max(min(right_speed, MAX_SPEED), -MAX_SPEED)
                
                set_motor_speed(left_speed, right_speed)

                # Visual Debugging
                cv2.drawContours(roi, [largest_contour], -1, (0, 255, 0), 2)
                cv2.circle(roi, (cx, 40), 5, (255, 0, 0), -1)
        
        else:
            # Line Lost Recovery
            print("Line Lost! Searching...")
            integral = 0 # Reset integral so it doesn't spin wildly
            
            # Simple spin search: rotate based on last known direction
            if 'error' in locals() and error > 0:
                set_motor_speed(60, -40)
            else:
                set_motor_speed(-40, 60)

        # Show Vision
        cv2.imshow("Robot Eye", roi)
        cv2.imshow("Threshold", thresh)

        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    stop_motors()
    pwm_left.stop()
    pwm_right.stop()
    GPIO.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()