import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import time
from collections import deque
 
# ============================================================
# 1. TFLITE RUNTIME SETUP
# ============================================================
try:
    import tflite_runtime.interpreter as tflite
    def make_interpreter(path):
        return tflite.Interpreter(model_path=path)
    print("[INFO] Using tflite-runtime")
except ImportError:
    import tensorflow as tf
    def make_interpreter(path):
        return tf.lite.Interpreter(model_path=path)
    print("[INFO] Using tensorflow")
 
# ============================================================
# 2. CONFIGURATION
# ============================================================
 
# -- Teachable Machine files ----------------------------------
TM_MODEL_PATH  = "/home/pi/pw3_tm/model_unquant.tflite"
TM_LABELS_PATH = "/home/pi/pw3_tm/labels.txt"
TM_INPUT_SIZE  = 224
 
# -- Detection tuning -----------------------------------------
CONF_THRESH     = 0.80
TM_EVERY_N      = 4
SMOOTH_WINDOW   = 5
SMOOTH_RATIO    = 0.70
SYMBOL_COOLDOWN = 3.0
BACKGROUND_CLASS = "background"
 
# -- Pre-filtering --------------------------------------------
USE_PREFILTER    = True
MIN_SYMBOL_AREA  = 400
MAX_SYMBOL_AREA  = 15000
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 4.0
 
# -- Motor pins (BOARD numbering) ----------------------------
ENA, IN1, IN2 = 15, 11, 13
ENB, IN3, IN4 = 22, 18, 16
 
# -- Speed settings ------------------------------------------
BASE_SPEED      = 33
MAX_SPEED       = 90
SLOW_SPEED      = 30
ROTATE_SPEED    = 75
ROTATE_DURATION = 2.0
TURN_DURATION   = 0.7   # duration for a 90-degree left/right turn
STOP_DURATION   = 3.0
 
# -- PI gains -------------------------------------------------
KP = 0.5
KI = 0.02
 
# -- Colour thresholds (HSV) ----------------------------------
YELLOW_LO = np.array([20, 100, 100])
YELLOW_HI = np.array([35, 255, 255])
RED_LO1   = np.array([0,  95,  95])
RED_HI1   = np.array([180, 255, 255])
RED_LO2   = np.array([120, 120,  70])
RED_HI2   = np.array([180, 255, 255])
COLOUR_MIN_PX   = 100
COLOUR_MIN_AREA = 80
 
# -- Line detection -------------------------------------------
BLACK_THRESH     = 60
MIN_LINE_AREA    = 500
MIN_XROAD_AREA   = 300
ROI_HEIGHT       = 80
LOST_SEARCH_SPEED = 50
 
# ============================================================
# 3. GPIO / PWM SETUP
# ============================================================
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
 
for pin in [ENA, IN1, IN2, ENB, IN3, IN4]:
    GPIO.setup(pin, GPIO.OUT)
 
pwm_left  = GPIO.PWM(ENA, 1000)
pwm_right = GPIO.PWM(ENB, 1000)
pwm_left.start(0)
pwm_right.start(0)

# ============================================================
# 4. MOTOR HELPERS
# ============================================================
def set_motor_speed(left, right):
    left  = max(min(int(left),  MAX_SPEED), -MAX_SPEED)
    right = max(min(int(right), MAX_SPEED), -MAX_SPEED)
 
    if left >= 0:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
    else:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
    pwm_left.ChangeDutyCycle(abs(left))
 
    if right >= 0:
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
    else:
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
    pwm_right.ChangeDutyCycle(abs(right))
 
def stop_motors():
    set_motor_speed(0, 0)
 
def rotate_360():
    """Blocking 360-degree spin for RECYCLE symbol."""
    print("[ACTION] RECYCLE -> rotating 360 degrees")
    set_motor_speed(-ROTATE_SPEED, ROTATE_SPEED)
    time.sleep(ROTATE_DURATION)
    stop_motors()
    time.sleep(0.15)
 
def turn_left():
    """Blocking left turn executed immediately on LEFT symbol detection."""
    print("[ACTION] LEFT -> turning left now")
    set_motor_speed(ROTATE_SPEED, -ROTATE_SPEED)
    time.sleep(TURN_DURATION)
    stop_motors()
    time.sleep(0.15)
 
def turn_right():
    """Blocking right turn executed immediately on RIGHT symbol detection."""
    print("[ACTION] RIGHT -> turning right now")
    set_motor_speed(ROTATE_SPEED, -ROTATE_SPEED)
    time.sleep(TURN_DURATION)
    stop_motors()
    time.sleep(0.15)
    
# ============================================================
# 5. TEACHABLE MACHINE DETECTOR
# ============================================================
class TMDetector:
    def __init__(self, model_path, labels_path):
        self.interpreter   = None
        self.input_details = None
        self.output_details = None
        self.labels   = []
        self.ready    = False
        self._smoother = deque(maxlen=SMOOTH_WINDOW)
        self._cooldown = {}
        self._inf_count = 0
 
        self._load_labels(labels_path)
        self._load_model(model_path)
 
    def _load_labels(self, path):
        try:
            with open(path, 'r') as f:
                lines = f.read().strip().splitlines()
            self.labels = []
            for line in lines:
                parts = line.strip().split(None, 1)
                name = parts[1].strip().lower() if (
                    len(parts) == 2 and parts[0].isdigit()
                ) else parts[0].strip().lower()
                self.labels.append(name)
            print(f"[INFO] Labels ({len(self.labels)}): {self.labels}")
            if BACKGROUND_CLASS not in self.labels:
                print(f"[WARNING] '{BACKGROUND_CLASS}' not in labels may get false positives")
        except Exception as e:
            print(f"[ERROR] Labels load failed: {e}")
 
    def _load_model(self, path):
        try:
            self.interpreter = make_interpreter(path)
            self.interpreter.allocate_tensors()
            self.input_details  = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]
            print(f"[INFO] Model loaded: {path}")
            self.ready = True
        except Exception as e:
            print(f"[ERROR] Model load failed: {e}")
 
    def has_potential_symbol(self, frame):
        if not USE_PREFILTER:
            return True
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if MIN_SYMBOL_AREA < area < MAX_SYMBOL_AREA:
                x, y, w, h = cv2.boundingRect(c)
                if h > 0 and MIN_ASPECT_RATIO < (w / h) < MAX_ASPECT_RATIO:
                    return True
        return False
 
    def classify(self, frame, use_prefilter=True):
        if not self.ready:
            return BACKGROUND_CLASS, 0.0
        h, w = frame.shape[:2]
        zone = frame[0:int(h * 0.70), :]
        if use_prefilter and not self.has_potential_symbol(zone):
            return BACKGROUND_CLASS, 0.0
        img = cv2.resize(zone, (TM_INPUT_SIZE, TM_INPUT_SIZE))
        if self.input_details['dtype'] == np.float32:
            tensor = (img.astype(np.float32) / 255.0)[np.newaxis]
        else:
            tensor = img.astype(np.uint8)[np.newaxis]
        self.interpreter.set_tensor(self.input_details['index'], tensor)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.output_details['index'])[0]
        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])
        label = self.labels[idx] if idx < len(self.labels) else f"cls{idx}"
        self._inf_count += 1
        if self._inf_count % 50 == 0:
            print(f"[TM] raw={label} conf={conf:.2f}")
        return label, conf
    def update_smooth(self, label, conf):
        self._smoother.append((label, conf))
        counts = {}
        for lbl, _ in self._smoother:
            counts[lbl] = counts.get(lbl, 0) + 1
        best = max(counts, key=counts.__getitem__)
        if counts[best] < len(self._smoother) * SMOOTH_RATIO:
            return BACKGROUND_CLASS, 0.0
        avg_conf = float(np.mean([c for l, c in self._smoother if l == best]))
        if best == BACKGROUND_CLASS:
            return BACKGROUND_CLASS, 0.0
        return best, avg_conf
 
    def in_cooldown(self, label):
        return time.time() - self._cooldown.get(label, 0) < SYMBOL_COOLDOWN

    def mark_acted(self, label):
        self._cooldown[label] = time.time()
        self._smoother.clear()
        print(f"[COOLDOWN] {label} cooldown started ({SYMBOL_COOLDOWN}s)")
 
# ============================================================
# 6. COLOUR LINE DETECTION
# ============================================================
def get_colour_cx(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, np.array([85, 80, 80]), np.array([110, 255, 255]))
    red = cv2.inRange(hsv, np.array([120, 180, 150]), np.array([170, 255, 255]))    
    
    y_px = cv2.countNonZero(yellow)
    r_px = cv2.countNonZero(red)
 
    if not hasattr(get_colour_cx, '_n'):
        get_colour_cx._n = 0
    get_colour_cx._n += 1
    if get_colour_cx._n % 30 == 0:
        print(f"[COLOUR] yellow_px={y_px}  red_px={r_px}")
 
    for colour, mask, px in [('yellow', yellow, y_px), ('red', red, r_px)]:
        if px < 100:
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        M = cv2.moments(max(cnts, key=cv2.contourArea))
        if M['m00'] > 80:
            return colour, int(M['m10'] / M['m00'])
    return None, None

# ============================================================
# 7. TWO-BLACK-LINE DETECTION
# ============================================================
def get_two_black_cx(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        return None, None
    top2 = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    cxs  = []
    for c in top2:
        M = cv2.moments(c)
        if M['m00'] > MIN_XROAD_AREA:
            cxs.append(int(M['m10'] / M['m00']))
    if len(cxs) < 2:
        return None, None
    cxs.sort()
    return cxs[0], cxs[1]
 
# ============================================================
# 8. HUD OVERLAY
# ============================================================
def draw_hud(frame, stable_label, stable_conf, raw_label, raw_conf,
             pending_dir, w, prefilter_status=True):
    y = 20
    if stable_label not in (BACKGROUND_CLASS, "nothing", "") and stable_conf >= CONF_THRESH:
        cv2.putText(frame, f"{stable_label.upper()} {stable_conf:.2f}",
                    (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 2)
        y += 30
    if raw_label != BACKGROUND_CLASS:
        cv2.putText(frame, f"raw: {raw_label} {raw_conf:.2f}",
                    (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1)
        y += 25
    if not prefilter_status:
        cv2.putText(frame, "PRE-FILTER: NO SYMBOL",
                    (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 200), 1)
        y += 20
    if pending_dir:
        cv2.putText(frame, f"QUEUED: {pending_dir.upper()}",
                    (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 0, 255), 2)
    cv2.putText(frame, f"THRESH: {CONF_THRESH}",
                (w - 110, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
 
# ============================================================
# 9. MAIN
# ============================================================
def main():
    detector = TMDetector(TM_MODEL_PATH, TM_LABELS_PATH)
 
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (320, 240), "format": "RGB888"}
    ))
    picam2.start()
    time.sleep(1.5)
 
    print("=" * 60)
    print("  Robot Car | pw2_verysuccess base + AI symbol detection")
    print(f"  BASE_SPEED={BASE_SPEED}  MAX_SPEED={MAX_SPEED}  KP={KP}  KI={KI}")
    print(f"  ROI bottom {ROI_HEIGHT}px  |  BLACK_THRESH={BLACK_THRESH}")
    print(f"  Confidence threshold: {CONF_THRESH}")
    print("  Press 'q' to quit, Ctrl-C to stop")
    print("=" * 60)
 
    # ---- State ----
    integral   = 0.0
    last_time  = time.time()
    last_error = 0
    slow_until = 0.0
    pending_dir = None  # kept for crossroad logic but no longer set by left/right symbols
    
    frame_count  = 0
    raw_label    = BACKGROUND_CLASS
    raw_conf     = 0.0
    stable_label = BACKGROUND_CLASS
    stable_conf  = 0.0
 
    try:
        while True:
            frame = picam2.capture_array()
            now   = time.time()
            dt    = now - last_time
            last_time = now
            frame_count += 1
 
            height, width, _ = frame.shape
 
            # ================================================
            # A. SYMBOL DETECTION (every N frames)
            # ================================================
            if frame_count % TM_EVERY_N == 0:
                raw_label, raw_conf = detector.classify(frame, use_prefilter=USE_PREFILTER)
 
            stable_label, stable_conf = detector.update_smooth(raw_label, raw_conf)
            top_half      = frame[0:int(height * 0.70), :]
            prefilter_ok  = detector.has_potential_symbol(top_half) if USE_PREFILTER else True
 
            # ================================================
            # B. ACT ON CONFIRMED SYMBOL
            # ================================================
            actionable = (
                stable_label not in (BACKGROUND_CLASS, "nothing", "")
                and stable_conf >= CONF_THRESH
                and not detector.in_cooldown(stable_label)
            )
 
            if actionable:
                print(f"\n[SYMBOL] {stable_label.upper()}  conf={stable_conf:.2f}")
                detector.mark_acted(stable_label)
 
                if stable_label in ("stop", "stop2"):
                    stop_motors()
                    integral = 0.0
                    print(f"[ACTION] STOP -> halting {STOP_DURATION}s")
                    time.sleep(STOP_DURATION)
 
                elif stable_label == "recycle":
                    rotate_360()
                    integral = 0.0
 
                # --- CHANGED: left/right now turn immediately, no queue ---
                elif stable_label == "left":
                    integral = 0.0
                    turn_left()
 
                elif stable_label == "right":
                    integral = 0.0
                    turn_right()
                    
                # -----------------------------------------------------------
 
                elif stable_label == "forward":
                    pending_dir = None
                    print("[ACTION] FORWARD (queue cleared)")
 
                elif stable_label in ("fingerprint", "qr"):
                    slow_until = now + 3.0
                    print(f"[ACTION] {stable_label.upper()} -> slowing 3s")
 
            # ================================================
            # C. BASE SPEED
            # ================================================
            base = SLOW_SPEED if now < slow_until else BASE_SPEED
 
            # ================================================
            # D. ROI BOTTOM 80px
            # ================================================
            roi = frame[height - ROI_HEIGHT:height, :]
 
            gray    = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, BLACK_THRESH, 255, cv2.THRESH_BINARY_INV)
 
            # ================================================
            # E. CROSSROAD TYPE 1: Colour line
            # ================================================
            colour, col_cx = get_colour_cx(roi)
            if colour is not None:
                error      = col_cx - (width // 2)
                integral   = 0.0
                correction = int(error * KP)
                set_motor_speed(base + correction, base - correction)
                last_error = error
 
                tag = "RED LINE" if colour == "red" else "YELLOW LINE"
                cv2.putText(frame, tag, (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 60, 255) if colour == "red" else (0, 200, 255), 2)
                draw_hud(frame, stable_label, stable_conf, raw_label, raw_conf,
                         pending_dir, width, prefilter_ok)
                cv2.imshow("Robot Eye", roi)
                cv2.imshow("Threshold", thresh)
                cv2.imshow("Robot View", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                continue
 
            # ================================================
            # F. CROSSROAD TYPE 2: Two black lines + arrow
            # ================================================
            left_cx, right_cx = get_two_black_cx(thresh)
            if left_cx is not None and right_cx is not None:
                if pending_dir == "left":
                    target_cx = left_cx
                elif pending_dir == "right":
                    target_cx = right_cx
                else:
                    target_cx = (left_cx + right_cx) // 2
 
                error      = target_cx - (width // 2)
                integral   = 0.0
                correction = int(error * KP)
                set_motor_speed(base + correction, base - correction)
                last_error = error
 
                dir_text = pending_dir if pending_dir else "STRAIGHT"
                cv2.putText(frame, f"TWO BLACK: {dir_text.upper()}", (5, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 0), 2)
                draw_hud(frame, stable_label, stable_conf, raw_label, raw_conf,
                         pending_dir, width, prefilter_ok)
                cv2.imshow("Robot Eye", roi)
                cv2.imshow("Threshold", thresh)
                cv2.imshow("Robot View", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                continue
            # ================================================
            # G. NORMAL LINE FOLLOWING
            # ================================================
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
 
                if M['m00'] > MIN_LINE_AREA:
                    cx = int(M['m10'] / M['m00'])
 
                    error     = cx - (width // 2)
                    integral += error * dt
                    integral = max(min(integral, 500), -500)
 
                    correction  = int((error * KP) + (integral * KI))
                    left_speed  = base + correction
                    right_speed = base - correction
 
                    left_speed  = max(min(left_speed,  MAX_SPEED), -MAX_SPEED)
                    right_speed = max(min(right_speed, MAX_SPEED), -MAX_SPEED)
 
                    set_motor_speed(left_speed, right_speed)
                    last_error = error
 
                    cv2.drawContours(roi, [largest_contour], -1, (0, 255, 0), 2)
                    cv2.circle(roi, (cx, 40), 5, (255, 0, 0), -1)
                else:
                    contours = []
 
            # ================================================
            # H. LINE LOST
            # ================================================
            if not contours:
                print("Line Lost! Searching...")
                integral = 0
                if last_error > 0:
                    set_motor_speed(60, -40)
                else:
                    set_motor_speed(-40, 60)
 
            # ================================================
            # I. DISPLAY
            # ================================================
            draw_hud(frame, stable_label, stable_conf, raw_label, raw_conf,
                     pending_dir, width, prefilter_ok)
 
            if now < slow_until:
                cv2.putText(frame, "SLOW", (width - 60, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
 
            cv2.imshow("Robot Eye",  roi)
            cv2.imshow("Threshold",  thresh)
            cv2.imshow("Robot View", frame)
 
            if cv2.waitKey(1) == ord('q'):
                break
 
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
 
    finally:
        stop_motors()
        pwm_left.stop()
        pwm_right.stop()
        GPIO.cleanup()
        picam2.stop()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")
 
# ============================================================
# 10. ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
