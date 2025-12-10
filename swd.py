# wand_tip_sender_pose20_opt.py
# Use MediaPipe *Pose* and send wand tip (u,v) in [0..1].
# Now supports LEFT / RIGHT hand selection + lower-res capture.

import cv2
import numpy as np
import socket, time, json
import mediapipe as mp

# ---------------- config ----------------
CAM_INDEX = 0

# Lower resolution for better performance
FRAME_W, FRAME_H = 640, 360

FLIP_HORIZONTAL = False                 # set True if you want a mirrored preview
SEND_ADDR = ("127.0.0.1", 5006)         # Unity receiver port (wand-only version)

TARGET_FPS = 30                         # send & preview cap
EMA_SEC    = 0.03                       # ~30 ms EMA on uv smoothing (set 0 to disable)
MIN_VIS    = 0.40                       # minimum landmark visibility to treat as valid

DRAW_TIP_DOT  = True
DRAW_SKELETON = False                   # you can turn this on to visualize Pose landmarks

# --- New: choose which hand to use for the "wand" ---
#   "right" -> use Right Index (landmark 20)
#   "left"  -> use Left  Index (landmark 19)
HAND = "right"   # <<< change this to "left" if you want left-hand wand

# ----------------------------------------

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

def to_uv(x:int, y:int, w:int, h:int):
    u = np.clip(x / float(w), 0.0, 1.0)
    v = np.clip(y / float(h), 0.0, 1.0)
    return float(u), float(v)

def main():
    # pick landmark index based on HAND
    if HAND.lower() == "left":
        lm_index = 19
        tip_label = "L-Index(19)"
    else:
        lm_index = 20
        tip_label = "R-Index(20)"

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # start auto; toggle with 'a'

    if not cap.isOpened():
        print("Could not open camera.")
        return

    pose = mp_pose.Pose(
        model_complexity=0,              # fast
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    prev_send = 0.0
    send_dt   = 1.0 / max(1, TARGET_FPS)
    uv_ema    = None

    print("Controls:")
    print("  q or ESC  : quit")
    print("  a         : toggle auto exposure")
    print("  [ / ]     : manual exposure down / up (forces manual mode)")
    print("  ; / '     : gain down / up")
    print(f"Using HAND = {HAND}  (landmark {lm_index})")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        tip_uv = None
        conf   = 0.0
        out    = frame.copy()

        # MediaPipe Pose landmark indices:
        # 19 = Left Index, 20 = Right Index
        if res and res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            h, w = frame.shape[:2]
            lm = lms[lm_index]
            if lm.visibility >= MIN_VIS:
                px, py = int(lm.x * w), int(lm.y * h)
                tip_uv = to_uv(px, py, w, h)
                conf   = float(np.clip(lm.visibility, 0.0, 1.0))

                if DRAW_TIP_DOT:
                    cv2.circle(out, (px, py), 10, (0, 255, 255), 2)
                    cv2.putText(out, tip_label, (px+8, py-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

            if DRAW_SKELETON:
                mp_drawing.draw_landmarks(
                    out, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )

        # EMA smoothing on uv (optional)
        if EMA_SEC > 0 and tip_uv is not None:
            # simple EMA with time-constant EMA_SEC
            dt    = 1.0 / max(1, TARGET_FPS)
            alpha = dt / (EMA_SEC + dt)
            if uv_ema is None:
                uv_ema = np.array(tip_uv, dtype=np.float32)
            else:
                uv_ema = (1.0 - alpha) * uv_ema + alpha * np.array(tip_uv, dtype=np.float32)
            tip_uv_smooth = (float(uv_ema[0]), float(uv_ema[1]))
        else:
            tip_uv_smooth = tip_uv

        # send at TARGET_FPS
        now = time.time()
        if now - prev_send >= send_dt:
            prev_send = now
            payload = {
                "uv": tip_uv_smooth if tip_uv_smooth else [None, None],
                "c":  float(conf),    # treat this as confidence/visibility
                "w":  FRAME_W,
                "h":  FRAME_H,
                "ts": now
            }
            try:
                sock.sendto(json.dumps(payload).encode("utf-8"), SEND_ADDR)
            except Exception:
                pass

        # HUD
        cv2.putText(out, f"hand:{HAND}  uv:{tip_uv_smooth if tip_uv_smooth else None}  vis:{conf:.2f}",
                    (12, FRAME_H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("wand tip sender (Pose Index)", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

        # exposure/gain hotkeys
        if key == ord('a'):
            auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            new = 0.25 if auto >= 0.75 else 0.75
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, new)
            print("AUTO_EXPOSURE ->", new)

        elif key == ord('['):  # darker (manual)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            v = cap.get(cv2.CAP_PROP_EXPOSURE)
            cap.set(cv2.CAP_PROP_EXPOSURE, v - 1)
            print("EXPOSURE ->", cap.get(cv2.CAP_PROP_EXPOSURE))

        elif key == ord(']'):  # brighter (manual)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            v = cap.get(cv2.CAP_PROP_EXPOSURE)
            cap.set(cv2.CAP_PROP_EXPOSURE, v + 1)
            print("EXPOSURE ->", cap.get(cv2.CAP_PROP_EXPOSURE))

        elif key == ord(';'):  # less gain
            v = cap.get(cv2.CAP_PROP_GAIN)
            cap.set(cv2.CAP_PROP_GAIN, max(0, v - 1))
            print("GAIN ->", cap.get(cv2.CAP_PROP_GAIN))

        elif key == ord("'"):  # more gain
            v = cap.get(cv2.CAP_PROP_GAIN)
            cap.set(cv2.CAP_PROP_GAIN, v + 1)
            print("GAIN ->", cap.get(cv2.CAP_PROP_GAIN))

    pose.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
