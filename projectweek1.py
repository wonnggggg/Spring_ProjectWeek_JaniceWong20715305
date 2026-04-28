from gpiozero import Motor
from time import sleep

# --- 1. Pin Configuration ---
# Right Motor
motor_right = Motor(forward=18, backward=16, enable=22)
# Left Motor
motor_left  = Motor(forward=11, backward=13, enable=15)

# --- 2. Calibration Data  ---
# We use this to calculate the ratio
MEASURED_PWM = 140
MEASURED_SPEED_CM_S = 29.7

# Calculate the Ratio: How much PWM is needed for 1 cm/s?
# Formula: Ratio = PWM / Speed
# 140 / 29.7 = approx 4.71
SPEED_RATIO = MEASURED_PWM / MEASURED_SPEED_CM_S

def get_duty_cycle_from_speed(target_speed):
    """
    Converts target speed (cm/s) to Duty Cycle (0.0 - 1.0)
    """
    # 1. Calculate required Arduino-style PWM (0-255)
    required_pwm = target_speed * SPEED_RATIO
    
    # 2. Safety Cap: Cannot exceed 255 (Max Speed)
    if required_pwm > 255:
        required_pwm = 255
        print(f"Warning: Speed too high! Capping at Max Speed.")
    
    # 3. Convert to 0.0-1.0 for gpiozero
    duty_cycle = required_pwm / 255
    
    print(f"Target: {target_speed} cm/s -> PWM: {int(required_pwm)} -> Duty: {duty_cycle:.2f}")
    return duty_cycle

# --- 3. Main Program ---

try:
    print(f"--- Calibration Info ---")
    print(f"Based on: PWM {MEASURED_PWM} = {MEASURED_SPEED_CM_S} cm/s")
    print("------------------------")

    # Ask user for input
    user_input = input("Enter target speed in cm/s (e.g., 15, 30, 40): ")
    target_speed = float(user_input)

    # Convert to motor power
    power = get_duty_cycle_from_speed(target_speed)

    print(f"Running at {target_speed} cm/s for 5 seconds...")
    
    # Run Motors
    motor_right.forward(power)
    motor_left.forward(power)
    
    sleep(5) # Run for 5 seconds

    print("Stop.")
    motor_right.stop()
    motor_left.stop()

except ValueError:
    print("Error: Please enter a valid number.")
except KeyboardInterrupt:
    print("Stopped by user.")