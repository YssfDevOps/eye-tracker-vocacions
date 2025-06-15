import time
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Tuple, Sequence
import cv2, mss
import numpy as np
from gaze   import Detector, Predictor
from models import FullModel
from utils  import get_config, clamp_value, plot_trajectory
import keyboard

SETTINGS, COLOURS, EYETRACKER, TF = get_config("config.ini")

def _weighted_avg(buf: Sequence[float], weights: np.ndarray) -> float:
    return float(np.sum(np.array(buf) * weights) / weights.sum())

def _setup_monitor(monitors: list[dict], idx: int) -> Tuple[dict, int, int]:
    mon = monitors[idx]
    w, h = mon["width"], mon["height"]
    monitor = {"top": mon["top"], "left": mon["left"], "width": w, "height": h}
    return monitor, w, h

def tracker():
    detector  = Detector(output_size=SETTINGS["image_size"])
    predictor = Predictor(
        FullModel,
        model_path = Path("trained_models/full/eyetracking_model.pt"),
        cfg_json   = Path("trained_models/full/eyetracking_config.json"),
        gpu        = 0,
    )
    screen_err = np.load("trained_models/full/errors.npy")

    # smoothing buffers
    win_pos   = SETTINGS["avg_window_length"]
    win_err   = win_pos * 2
    track_x   = deque([0]*win_pos, maxlen=win_pos)
    track_y   = deque([0]*win_pos, maxlen=win_pos)
    track_err = deque([0]*win_err, maxlen=win_err)
    w_pos     = np.arange(1, win_pos+1)
    w_err     = np.arange(1, win_err+1)

    is_recording = False
    record_pending = False
    traj_points = []
    screenshot_img = None

    with mss.mss() as sct:
        monitor, scr_w, scr_h = _setup_monitor(sct.monitors, EYETRACKER["monitor_num"])

        def on_toggle():
            nonlocal record_pending
            record_pending = True
        keyboard.add_hotkey('r', on_toggle)

        writer = None
        if EYETRACKER["write_to_disk"]:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_sz = (int(scr_w*EYETRACKER["screen_capture_scale"]),
                      int(scr_h*EYETRACKER["screen_capture_scale"]))
            writer = cv2.VideoWriter(f"media/recordings/{dt_str}.mp4",
                                     fourcc,
                                     EYETRACKER["tracker_frame_rate"],
                                     out_sz)

        last = time.time()

        """Delete this part before sending code"""
        cv2.namedWindow("Eye-tracker", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            "Eye-tracker",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        """---------------------------"""

        while True:
            if cv2.waitKey(1) & 0xFF == 27:
                break

            if record_pending:
                record_pending = False
                if not is_recording:
                    is_recording = True
                    traj_points = []
                    screenshot_img = np.array(sct.grab(monitor))
                    print("[*] Started recording trajectory")
                else:
                    is_recording = False
                    out_path = plot_trajectory(screenshot_img, traj_points)
                    print(f"[+] Trajectory saved in {out_path}")


            now = time.time()
            if now - last < 1/EYETRACKER["tracker_frame_rate"]:
                continue
            fps = 1 / (now - last)
            last = now

            l_eye, r_eye, face, face_al, head_pos, head_ang = detector.get_frame()
            x_pred, y_pred = predictor.predict(face_al, l_eye, r_eye,
                                               head_pos, head_angle=head_ang)

            track_x.append(x_pred)
            track_y.append(y_pred)
            x_cl = clamp_value(int(x_pred), scr_w-1)
            y_cl = clamp_value(int(y_pred), scr_h-1)
            track_err.append(screen_err[x_cl, y_cl]*.75)

            x_vis = _weighted_avg(track_x, w_pos)
            y_vis = _weighted_avg(track_y, w_pos)
            x_vis = clamp_value(int(x_vis), scr_w-1)
            y_vis = clamp_value(int(y_vis), scr_h-1)
            rad   = _weighted_avg(track_err, w_err)

            if is_recording:
                traj_points.append((x_vis, y_vis))

            frame   = np.array(sct.grab(monitor))
            overlay = frame.copy()
            centre  = (int(x_vis), int(y_vis))

            cv2.circle(overlay, centre, int(rad), (255,255,255,60), -1)
            cv2.circle(frame,   centre, int(rad), COLOURS["green"], 4)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            text = "Currently recording trajectory! Press R to stop!" if is_recording else "Press R to record trajectory"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = (frame.shape[1] - tw) // 2
            ty = frame.shape[0] - 20
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOURS["green"], 2)

            cv2.putText(frame, f"fps {fps:5.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOURS["green"], 2)
            cv2.putText(frame, f"({x_pred:4.0f},{y_pred:4.0f})", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS["green"], 2)

            frame = cv2.resize(frame,
                               (int(scr_w*EYETRACKER["screen_capture_scale"]),
                                int(scr_h*EYETRACKER["screen_capture_scale"])))
            cv2.imshow("Eye-tracker", frame)
            if writer: writer.write(frame)

        if writer: writer.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker()
