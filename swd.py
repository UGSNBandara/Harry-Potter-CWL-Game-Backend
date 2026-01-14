# wand_tip_sender_pose20_opt.py
# Use MediaPipe *Pose* and send wand tip (u,v) in [0..1].
# Now supports LEFT / RIGHT hand selection + lower-res capture.
# And can switch hand at runtime via small UDP JSON from Unity.

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

# --- New: runtime hand config from Unity ---
#   default used at start; Unity can change it via UDP
HAND = "right"   # "right" or "left"

# Unity → Python hand-config UDP
CONFIG_LISTEN_ADDR = ("0.0.0.0", 5011)  # Unity will send to 127.0.0.1:5011

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
    global HAND

    # extra socket to receive small JSON config from Unity
    config_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    config_sock.bind(CONFIG_LISTEN_ADDR)
    config_sock.setblocking(False)  # non-blocking

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
    print("  Ctrl+C    : quit")
    print(f"Initial HAND = {HAND}  (right=R-Index(20), left=L-Index(19))")
    print(f"Listening for hand config on {CONFIG_LISTEN_ADDR} (JSON: {{\"hand\":\"left\"}} / {{\"hand\":\"right\"}})")

    try:
        while True:
            # --- check if Unity sent a hand update ---
            try:
                data, addr = config_sock.recvfrom(1024)
                msg = json.loads(data.decode("utf-8"))
                new_hand = msg.get("hand", "").lower()
                if new_hand in ("left", "right") and new_hand != HAND:
                    HAND = new_hand
                    print(f"[wand_tip] Switched HAND to {HAND}")
            except BlockingIOError:
                # no config received, keep going
                pass
            except Exception as e:
                print("[wand_tip] Config parse error:", e)

            ok, frame = cap.read()
            if not ok:
                break

            # Rate limiting: only process if enough time has passed
            now = time.time()
            if now - prev_send < send_dt:
                time.sleep(0.005)
                continue
            
            prev_send = now

            if FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            # decide landmark index based on current HAND
            if HAND.lower() == "left":
                lm_index = 19
                tip_label = "L-Index(19)"
            else:
                lm_index = 20
                tip_label = "R-Index(20)"

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            tip_uv = None
            conf   = 0.0

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
                        cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)
                        cv2.putText(frame, tip_label, (px+8, py),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            # EMA smoothing on uv (optional)
            if EMA_SEC > 0 and tip_uv is not None:
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

            # optional preview
            if DRAW_SKELETON and res and res.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )

            cv2.imshow("Pose Wand Tip", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
                
    except KeyboardInterrupt:
        pass

    pose.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
