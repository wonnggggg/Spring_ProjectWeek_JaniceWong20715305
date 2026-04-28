import time
import os
import cv2
import numpy as np
from picamera2 import Picamera2

# ---------------- Camera setup ----------------
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

print("Symbol Detection + Template Matching")
print("Keys:  t = save detected symbol as template,  q = quit")

# ===================== SETTINGS =====================
ROI_START = 0.60

# ===================== TEMPLATE SAVE SETTINGS =====================
SAVE_DIR = "templates"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== SYMBOL DETECTION =====================
def preprocess_template(crop_gray, size=(120, 120)):
    crop_eq = cv2.equalizeHist(crop_gray)
    h, w = crop_eq.shape
    diff = abs(h - w)
    top, bottom, left, right = 0, 0, 0, 0
    if h > w:
        left = diff // 2
        right = diff - left
    elif w > h:
        top = diff // 2
        bottom = diff - top

    crop_sq = cv2.copyMakeBorder(crop_eq, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
    crop_pad = cv2.copyMakeBorder(crop_sq, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    tpl = cv2.resize(crop_pad, size, interpolation=cv2.INTER_AREA)
    return tpl

def load_templates():
    tpls = {}
    for f in sorted(os.listdir(SAVE_DIR)):
        if f.lower().endswith(".png"):
            path = os.path.join(SAVE_DIR, f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                tpls[f] = img
    return tpls

templates = load_templates()
print(f"Loaded {len(templates)} templates from '{SAVE_DIR}'")

MATCH_THRESH = 0.73
COOLDOWN = 1.0
last_print = 0.0

def detect_and_crop_symbol(frame_rgb):
    H, W, _ = frame_rgb.shape
    roi_bottom = int(H * ROI_START)
    roi = frame_rgb[0:roi_bottom, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    bin_inv = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 81, 8)

    k = np.ones((5, 5), np.uint8)
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, k)
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(bin_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, bin_inv, None

    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if 1000 < area < (W * roi_bottom * 0.6):
            x, y, w, h = cv2.boundingRect(c)
            margin = 5
            if x > margin and y > margin and (x + w) < (W - margin) and (y + h) < (roi_bottom - margin):
                if 0.25 < (w / float(h)) < 4.0:
                    hull = cv2.convexHull(c)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                        if solidity > 0.35:
                            valid_contours.append(c)

    if not valid_contours:
        return None, None, bin_inv, None

    c = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    pad = 12
    rx0 = max(0, x - pad)
    ry0 = max(0, y - pad)
    rx1 = min(W, x + w + pad)
    ry1 = min(roi_bottom, y + h + pad)

    crop_gray = gray[ry0:ry1, rx0:rx1]
    symbol_box_full = (rx0, ry0, rx1, ry1)

    return crop_gray, symbol_box_full, bin_inv, c

# ===================== GEOMETRY CLASSIFIER =====================
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
# ===================== MAIN LOOP =====================
try:
    while True:
        frame = picam2.capture_array()
        h, w, _ = frame.shape
        disp = frame.copy()

        # ---------------- SYMBOL DETECTION & TEMPLATE MATCHING ----------------
        crop_gray, symbol_box, bin_sym, symbol_contour = detect_and_crop_symbol(frame)

        det_name = None
        det_score = 0.0
        geo_label = None

        if crop_gray is not None:
            processed_crop = preprocess_template(crop_gray, size=(120, 120))
            geo_label = classify_geometry_from_contour(symbol_contour)

            if templates:
                for name, tpl in templates.items():
                    res = cv2.matchTemplate(processed_crop, tpl, cv2.TM_CCOEFF_NORMED)
                    score = res[0][0]
                    if score > MATCH_THRESH and score > det_score:
                        det_score = score
                        det_name = name

        # ---------------- DISPLAY ----------------
        roi_y = int(h * ROI_START)
        cv2.line(disp, (0, roi_y), (w, roi_y), (255, 0, 0), 2)
        cv2.putText(disp, "Symbol Search Area", (5, roi_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if symbol_box is not None:
            x0, y0, x1, y1 = symbol_box
            if det_name is not None:
                display_name = det_name.split('.')[0]
                if '_' in display_name and display_name.rsplit('_', 1)[1].isdigit():
                    display_name = display_name.rsplit('_', 1)[0]

                cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 3)
                cv2.putText(disp, f"{display_name} ({det_score:.2f})", (x0, max(20, y0 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                now = time.time()
                if now - last_print > COOLDOWN:
                    print(f"Detected: {display_name} (score={det_score:.2f})")
                    last_print = now
            else:
                cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.putText(disp, "Unknown (press 't')", (x0, max(20, y0 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if geo_label:
                cv2.putText(disp, f"Geo: {geo_label}", (x0, y1 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Camera", disp)
        if bin_sym is not None:
            cv2.imshow("Binary (Symbol)", bin_sym)
        
        # ---------------- KEYBOARD INTERACTION ----------------
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('t'):
            if crop_gray is None:
                print("No clear symbol detected to save.")
            else:
                name_input = ""
                while True:
                    prompt_disp = disp.copy()
                    cv2.rectangle(prompt_disp, (20, h//2 - 60), (w-20, h//2 + 40), (0, 0, 0), -1)
                    cv2.rectangle(prompt_disp, (20, h//2 - 60), (w-20, h//2 + 40), (255, 255, 255), 2)
                    cv2.putText(prompt_disp, "TYPE SYMBOL NAME (e.g. UP, DOWN):", (30, h//2 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(prompt_disp, f"> {name_input}_", (30, h//2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    cv2.putText(prompt_disp, "Press ENTER to save, ESC to cancel.", (30, h//2 + 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.imshow("Camera", prompt_disp)

                    k = cv2.waitKey(0) & 0xFF
                    if k == 13 or k == 10:
                        break
                    elif k == 27:
                        name_input = ""
                        break
                    elif k == 8 or k == 127:
                        name_input = name_input[:-1]
                    else:
                        if chr(k).isalnum() or k in [ord('-'), ord('_')]:
                            name_input += chr(k).upper()

                if name_input.strip() != "":
                    idx = 1
                    while os.path.exists(os.path.join(SAVE_DIR, f"{name_input}_{idx}.png")):
                        idx += 1
                    tpl = preprocess_template(crop_gray, size=(120, 120))
                    filename = os.path.join(SAVE_DIR, f"{name_input}_{idx}.png")
                    cv2.imwrite(filename, tpl)
                    print(f"Saved: {filename}")
                    templates = load_templates()
                    print(f"Now loaded {len(templates)} templates")

        elif key == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    picam2.stop()