import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import time
import os

# ============================================================
#  PIN & SPEED CONFIGURATION  (from pw2_verysuccess.py)
# ============================================================
ENA, IN1, IN2 = 15, 11, 13
ENB, IN3, IN4 = 22, 18, 16

BASE_SPEED = 30
MAX_SPEED  = 90
KP = 0.4          # Proportional gain
KI = 0.02         # Integral gain

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

for pin in [ENA, IN1, IN2, ENB, IN3, IN4]:
    GPIO.setup(pin, GPIO.OUT)

pwm_left  = GPIO.PWM(ENA, 1000)
pwm_right = GPIO.PWM(ENB, 1000)
pwm_left.start(0)
pwm_right.start(0)

# ============================================================
#  MOTOR HELPERS
# ============================================================
def set_motor_speed(left, right):
    if left >= 0:
        GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    else:
        GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
    pwm_left.ChangeDutyCycle(min(abs(left), 100))

    if right >= 0:
        GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    else:
        GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
    pwm_right.ChangeDutyCycle(min(abs(right), 100))

def stop_motors():
    set_motor_speed(0, 0)

# ============================================================
#  CAMERA SETUP
# ============================================================
import threading

class ThreadedCamera:
    """Continuously grabs the latest frame in a background thread.
    The main loop always gets the most recent frame instantly."""
    def __init__(self, picam):
        self.picam   = picam
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        self.thread  = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            f = self.picam.capture_array()
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1.0)   # Let camera warm up

cam = ThreadedCamera(picam2)


# ============================================================
#  TEMPLATE LOADING  (from pw2_shaperecognize.py)
# ============================================================
SAVE_DIR     = "templates"
MATCH_THRESH = 0.73
os.makedirs(SAVE_DIR, exist_ok=True)

def preprocess_template(crop_gray, size=(120, 120)):
    crop_eq = cv2.equalizeHist(crop_gray)
    h, w = crop_eq.shape
    diff = abs(h - w)
    top = bottom = left = right = 0
    if h > w:
        left  = diff // 2;  right  = diff - left
    elif w > h:
        top   = diff // 2;  bottom = diff - top
    crop_sq  = cv2.copyMakeBorder(crop_eq, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=255)
    crop_pad = cv2.copyMakeBorder(crop_sq, 10, 10, 10, 10,
                                   cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(crop_pad, size, interpolation=cv2.INTER_AREA)

def load_templates():
    tpls = {}
    for f in sorted(os.listdir(SAVE_DIR)):
        if f.lower().endswith(".png"):
            path = os.path.join(SAVE_DIR, f)
            img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                tpls[f] = img
    return tpls

templates = load_templates()
print(f"Loaded {len(templates)} templates from '{SAVE_DIR}'")

# ============================================================
#  SYMBOL DETECTION  (from pw2_shaperecognize.py)
# ============================================================
SYMBOL_ROI_END = 0.60   # Top 60 % of frame is the symbol search area

# Symbol detection runs every N frames to reduce CPU load.
# Line following always runs every frame.
SYMBOL_CHECK_EVERY = 3   # Check for symbols every 3 frames
_frame_count = 0

def detect_and_crop_symbol(frame_rgb):
    H, W, _ = frame_rgb.shape
    roi_bottom = int(H * SYMBOL_ROI_END)
    roi  = frame_rgb[0:roi_bottom, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    bin_inv = cv2.adaptiveThreshold(blur, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 81, 8)
    k = np.ones((5, 5), np.uint8)
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN,  k)
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(bin_inv, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, bin_inv, None

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (1000 < area < W * roi_bottom * 0.6):
            continue
        x, y, w, h = cv2.boundingRect(c)
        margin = 5
        if not (x > margin and y > margin and
                x + w < W - margin and y + h < roi_bottom - margin):
            continue
        if not (0.25 < w / float(h) < 4.0):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area > 0 and (area / hull_area) > 0.35:
            valid.append(c)

    if not valid:
        return None, None, bin_inv, None

    c = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    pad = 12
    rx0, ry0 = max(0, x - pad), max(0, y - pad)
    rx1, ry1 = min(W, x + w + pad), min(roi_bottom, y + h + pad)

    return gray[ry0:ry1, rx0:rx1], (rx0, ry0, rx1, ry1), bin_inv, c

# ============================================================
#  GEOMETRY CLASSIFIER  (exact copy from pw2_shaperecognize.py)
# ============================================================
def _order_pts(pts_approx):
    p = pts_approx.reshape(-1, 2).astype(np.float32)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    return p[np.argsort(ang)]

def _side_lengths(p4):
    d = []
    for i in range(4):
        x1, y1 = p4[i]
        x2, y2 = p4[(i + 1) % 4]
        d.append(float(np.hypot(x2 - x1, y2 - y1)))
    return d

def classify_geometry_from_contour(c):
    area = cv2.contourArea(c)
    if area < 1500: return None

    perim = cv2.arcLength(c, True)
    if perim < 1e-6: return None

    approx = cv2.approxPolyDP(c, 0.015 * perim, True)
    verts = len(approx)

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull) + 1e-6
    solidity = area / hull_area

    circ = 4 * np.pi * area / (perim * perim)

    if solidity < 0.80 and verts >= 8: return "STAR"
    if 7 <= verts <= 9 and solidity > 0.92 and circ < 0.85: return "OCTAGON"
    if 0.60 < circ < 0.85 and solidity < 0.90: return "PARTIAL_CIRCLE"
    if circ > 0.86 and solidity > 0.95 and verts >= 12: return "CIRCLE"

    if verts == 4 and solidity > 0.90:
        p = _order_pts(approx)
        s0, s1, s2, s3 = _side_lengths(p)

        def rel_diff(a, b): return abs(a - b) / max(a, b, 1e-6)
        opp0 = rel_diff(s0, s2)
        opp1 = rel_diff(s1, s3)

        if opp0 < 0.18 and opp1 < 0.18:
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            if 0.9 < ar < 1.1: return "DIAMOND"
            return "PARALLELOGRAM"

        if (opp0 < 0.18) ^ (opp1 < 0.18): return "TRAPEZOID"
        return "QUAD"

    return None

# ============================================================
#  SYMBOL IDENTIFICATION  (template match + geometry fallback)
# ============================================================
def identify_symbol(crop_gray, symbol_contour):
    """Returns (name_string, score_float)."""
    processed = preprocess_template(crop_gray, size=(120, 120))

    best_name, best_score = None, 0.0
    for name, tpl in templates.items():
        score = cv2.matchTemplate(processed, tpl, cv2.TM_CCOEFF_NORMED)[0][0]
        if score > MATCH_THRESH and score > best_score:
            best_score = score
            best_name  = name

    # Pretty-print template name: "UP_1.png" -> "UP"
    if best_name:
        label = best_name.split('.')[0]
        if '_' in label and label.rsplit('_', 1)[1].isdigit():
            label = label.rsplit('_', 1)[0]
        return label, best_score

    # Fall back to geometry
    if symbol_contour is not None:
        geo = classify_geometry_from_contour(symbol_contour)
        if geo:
            return geo, 0.0

    return "UNKNOWN", 0.0

# ============================================================
#  STOP-AND-IDENTIFY ROUTINE
# ============================================================
STOP_DURATION  = 10.0    # seconds to pause at symbol
SCAN_FRAMES    = 20     # frames to collect votes while stopped
COOLDOWN       = 3.0    # seconds before next detection triggers stop
last_stop_time = 0.0

def stop_and_identify(frame_rgb, disp):
    """Stop motors, scan several frames, vote on best label, announce."""
    stop_motors()
    print("\n Symbol detected - stopping to identify")
    cv2.putText(disp, "IDENTIFYING ...", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imshow("Robot Eye", disp)
    cv2.waitKey(1)

    votes = {}
    for _ in range(SCAN_FRAMES):
        f = picam2.capture_array()
        crop, _, _, contour = detect_and_crop_symbol(f)
        if crop is not None:
            label, score = identify_symbol(crop, contour)
            votes[label] = votes.get(label, 0) + (score if score > 0 else 0.5)
        time.sleep(0.05)

    if votes:
        winner = max(votes, key=votes.get)
        print(f"Symbol identified as: [{winner}]  (confidence votes={votes})")
    else:
        winner = "UNKNOWN"
        print("Could not identify symbol.")

    # Show result on screen for a moment
    result_frame = cam.read()
    if result_frame is None:
        result_frame = disp
    cv2.putText(result_frame, f"SYMBOL: {winner}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    cv2.imshow("Robot Eye", result_frame)
    cv2.waitKey(int(STOP_DURATION * 1000))

    return winner

# ============================================================
#  MAIN LOOP
# ============================================================
integral   = 0
last_time  = time.time()
last_error = 0
last_winner = None

print("Robot started - line following + symbol detection active.")
print("Press 'q' to quit.")

try:
    while True:
        frame = cam.read()
        if frame is None:
            continue
        H, W, _ = frame.shape
        disp = frame.copy()

        # -- Timing --------------------------------------------
        now = time.time()
        dt  = now - last_time
        last_time = now

        # -- LINE FOLLOW  (bottom 80 px ROI) -------------------
        roi   = frame[H - 80:H, :]
        gray  = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

        contours_line, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        if contours_line:
            lc = max(contours_line, key=cv2.contourArea)
            M  = cv2.moments(lc)
            if M['m00'] > 500:
                cx = int(M['m10'] / M['m00'])

                # PI control
                error     = cx - (W // 2)
                integral += error * dt
                integral  = max(min(integral, 500), -500)   # anti-windup
                correction = int(error * KP + integral * KI)

                left_spd  = max(min(BASE_SPEED + correction, MAX_SPEED), -MAX_SPEED)
                right_spd = max(min(BASE_SPEED - correction, MAX_SPEED), -MAX_SPEED)
                set_motor_speed(left_spd, right_spd)
                last_error = error

                # Draw line centroid
                cv2.circle(roi, (cx, 40), 6, (255, 0, 0), -1)
                cv2.drawContours(roi, [lc], -1, (0, 255, 0), 2)
            else:
                stop_motors()
        else:
            # Line lost - spin to search
            print("Line lost! Searching...")
            integral = 0
            if last_error > 0:
                set_motor_speed(60, -40)
            else:
                set_motor_speed(-40, 60)

        # -- SYMBOL DETECTION  (top 60 % ROI) ------------------
        _frame_count += 1
        crop_gray, sym_box, bin_sym, sym_contour = None, None, None, None
        if _frame_count % SYMBOL_CHECK_EVERY == 0:
            crop_gray, sym_box, bin_sym, sym_contour = detect_and_crop_symbol(frame)
            if crop_gray is None:
                last_winner = None  # clear label when no symbol visible

        if crop_gray is not None and (now - last_stop_time) > COOLDOWN:
            # Symbol found - stop and identify
            winner = stop_and_identify(frame, disp)
            last_winner = winner
            last_stop_time = time.time()

            # Resume PI state cleanly
            integral  = 0
            last_time = time.time()


        # -- DISPLAY -------------------------------------------
        roi_y = int(H * SYMBOL_ROI_END)
        cv2.line(disp, (0, roi_y), (W, roi_y), (255, 0, 0), 2)
        cv2.putText(disp, "Top Symbol Searching Area", (5, roi_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if sym_box is not None:
            x0, y0, x1, y1 = sym_box
            geo_label = classify_geometry_from_contour(sym_contour) if sym_contour is not None else None

            if last_winner is not None:
                cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 3)
                cv2.putText(disp, f"{last_winner}", (x0, max(20, y0 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.putText(disp, "Symbol detected...", (x0, max(20, y0 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if geo_label:
                cv2.putText(disp, f"Geo: {geo_label}", (x0, y1 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Robot Eye", disp)
        cv2.imshow("Threshold (Line)", thresh)
        if bin_sym is not None:
            cv2.imshow("Binary (Symbol)", bin_sym)

        if cv2.waitKey(1) & 0xFF == ord('q'):
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
    print("Robot stopped cleanly.")