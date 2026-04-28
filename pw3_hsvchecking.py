import cv2
import numpy as np
import pickle
import os
import time
from picamera2 import Picamera2
 
MODEL_FILE = "/home/pi/orb_symbol_model_v2.pkl"
 
# ============================================================
# CHECK 1 Model file contents
# ============================================================
print("\n" + "="*55)
print("CHECK 1: MODEL FILE")
print("="*55)
 
if not os.path.exists(MODEL_FILE):
    print(f"[FAIL] File not found: {MODEL_FILE}")
    print(" Train your model first!")
else:
    with open(MODEL_FILE, "rb") as f:
        data = pickle.load(f)
 
    print(f"[OK]   File found: {MODEL_FILE}")
    print(f"       Top-level type: {type(data)}")
 
    # Unwrap if nested
    if isinstance(data, dict) and 'symbols' in data:
        db = data['symbols']
        print("       Wrapper format: {'symbols': ...}")
    elif isinstance(data, dict):
        db = data
        print("       Format: plain dict")
    else:
        db = None
        print(f"[FAIL] Unexpected format: {type(data)}")
 
    if db is not None:
        print(f"\n       Number of symbol classes: {len(db)}")
        print(f"       Symbol names: {list(db.keys())}")
        print()
        for name, descs in db.items():
            print(f"       '{name}': {len(descs)} descriptor(s), ", end="")
            if descs:
                d = descs[0]
                if d is not None:
                    print(f"shape={d.shape}, dtype={d.dtype}")
                else:
                    print("descriptor is None  <-- PROBLEM")
            else:
                print("empty list  <-- PROBLEM")
# ============================================================
# CHECK 2 ORB keypoint detection on live camera
# ============================================================
print("\n" + "="*55)
print("CHECK 2: CAMERA + ORB KEYPOINTS")
print("="*55)
 
picam2 = Picamera2()
cfg = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"}
)
picam2.configure(cfg)
picam2.start()
time.sleep(1.5)
 
orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8,
                     edgeThreshold=31, WTA_K=2,
                     scoreType=cv2.ORB_HARRIS_SCORE,
                     patchSize=31, fastThreshold=20)
 
print("Camera started. Hold a symbol card in front of the camera.")
print("Watch the 'Keypoints' window you should see GREEN dots on the symbol.")
print("Press 's' to save snapshot, 'q' to continue to next check.\n")
 
snap_count = 0
while True:
    frame = picam2.capture_array()
    h, w  = frame.shape[:2]
 
    # Same ROI as main code top 55%
    roi_sym = frame[0 : int(h * 0.55), :]
    gray    = cv2.cvtColor(roi_sym, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    gray_bl = cv2.GaussianBlur(gray_eq, (3, 3), 0)
 
    kp, des = orb.detectAndCompute(gray_bl, None)
 
    # Draw keypoints
    kp_img = cv2.drawKeypoints(roi_sym, kp, None,
                               color=(0, 255, 0),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
    count_text = f"Keypoints: {len(kp)}"
    color = (0, 255, 0) if len(kp) >= 10 else (0, 0, 255)
    cv2.putText(kp_img, count_text, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
 
    if len(kp) < 10:
        cv2.putText(kp_img, "TOO FEW KEYPOINTS", (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
 
    # Draw the ROI boundary on full frame
    disp = frame.copy()
    cv2.rectangle(disp, (0, 0), (w, int(h * 0.55)), (255, 255, 0), 2)
    cv2.putText(disp, "Symbol ROI", (5, int(h * 0.55) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
 
    cv2.imshow("Full Frame (yellow box = symbol ROI)", disp)
    cv2.imshow("Keypoints", kp_img)
 
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        fname = f"/home/pi/diag_snap_{snap_count}.jpg"
        cv2.imwrite(fname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"[SAVED] {fname}  |  keypoints={len(kp)}")
        snap_count += 1
    elif key == ord('q'):
        break
 
cv2.destroyAllWindows()

# ============================================================
# CHECK 3 Yellow / Red HSV range on live camera
# ============================================================
print("\n" + "="*55)
print("CHECK 3: YELLOW / RED LINE COLOUR DETECTION")
print("="*55)
print("Place the yellow/red line under the camera (bottom of frame).")
print("The 'Yellow mask' and 'Red mask' windows should show WHITE pixels on the line.")
print("If masks are all black, the HSV range needs adjusting.")
print("Press 's' to print the HSV value at the centre of the frame.")
print("Press 'q' to quit.\n")
 
while True:
    frame = picam2.capture_array()
    h, w  = frame.shape[:2]
 
    # Same ROI as line follower bottom 80px
    roi_line = frame[h - 80 : h, :]
    hsv      = cv2.cvtColor(roi_line, cv2.COLOR_BGR2HSV)
 
    # Yellow actual camera reads H=97 (green-yellow due to Pi camera white balance)
    yellow_mask = cv2.inRange(hsv,
                              np.array([85, 150, 100]),
                              np.array([110, 255, 255]))
 
    # --- Red mask ---
    r1 = cv2.inRange(hsv, np.array([0,  120, 70]),  np.array([10,  255, 255]))
    r2 = cv2.inRange(hsv, np.array([110, 120, 70]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(r1, r2)
 
    # Pixel counts
    y_px = cv2.countNonZero(yellow_mask)
    r_px = cv2.countNonZero(red_mask)
 
    # Coloured overlay on ROI copy
    disp_roi = roi_line.copy()
    disp_roi[yellow_mask > 0] = [0, 255, 255]   # paint yellow pixels cyan
    disp_roi[red_mask > 0]  = [0, 0, 255]   # paint red pixels blue (BGR display)
 
    # Annotate
    y_col = (0, 200, 0) if y_px > 400 else (0, 0, 200)
    r_col = (0, 200, 0) if r_px > 400 else (0, 0, 200)
    cv2.putText(disp_roi, f"Yellow px: {y_px}", (2, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, y_col, 1)
    cv2.putText(disp_roi, f"Red px:    {r_px}", (2, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, r_col, 1)
 
    if y_px > 400:
        cv2.putText(disp_roi, "YELLOW DETECTED OK", (2, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    if r_px > 400:
        cv2.putText(disp_roi, "RED DETECTED OK", (2, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
 
    cv2.imshow("Line ROI colour overlay", disp_roi)
    cv2.imshow("Yellow mask", yellow_mask)
    cv2.imshow("Red mask",    red_mask)
 
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Print HSV value at frame centre tells you the exact hue of your line
        cy, cx = h // 2, w // 2
        full_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hval = full_hsv[cy, cx]
        print(f"[HSV at centre pixel ({cx},{cy})] H={hval[0]}  S={hval[1]}  V={hval[2]}")
        print("  Yellow range should be: H=20-35, S>100, V>100")
        print("  If your H value is outside 20-35, tell me the H value and I will fix the range.")
    elif key == ord('q'):
        break
 
cv2.destroyAllWindows()
picam2.stop()
print("\n[DONE] Diagnostics complete.")
print("Share the output above and I can tell you exactly what to fix.")
