import os
import sys
import csv
import cv2
import random
import pygame
import numpy as np
from scipy.stats import beta
from pygame.locals import *

from Target import Target
from Gaze import Detector
from utils import (
    get_config,
    bgr_to_rgb,
    clamp_value,
    plot_region_map,
    get_calibration_zones,
    get_undersampled_region,
)

# Read config.ini file
SETTINGS, COLOURS, EYETRACKER, TF = get_config("config.ini")

# Setup directories for saving images
data_dirs = ("data/l_eye", "data/r_eye", "data/face", "data/face_aligned", "data/head_pos")
for d in data_dirs:
    if not os.path.exists(d):
        os.makedirs(d)

# Setup CSV file to record labels
data_file_path = "data/positions.csv"
data_file_exists = os.path.isfile(data_file_path)
data_file = open(data_file_path, "a", newline="")
csv_writer = csv.writer(data_file, delimiter=",")
if not data_file_exists:
    csv_writer.writerow(["id", "x", "y", "head_angle"])

# Initialize pygame
pygame.init()
pygame.mouse.set_visible(0)
font_normal = pygame.font.SysFont(None, 30)
font_small = pygame.font.SysFont(None, 20)
pygame.display.set_caption("Calibrate and Collect")
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
w, h = pygame.display.get_surface().get_size()
center = (w // 2, h // 2)
webcam_surface = pygame.Surface((SETTINGS["image_size"] * 2, SETTINGS["image_size"] * 2))
calibration_zones = get_calibration_zones(w, h, SETTINGS["target_radius"])

# Create target and detector objects
target = Target(center, speed=SETTINGS["target_speed"], radius=SETTINGS["target_radius"])
detector = Detector(output_size=SETTINGS["image_size"])

# Load or initialize region map (for undersampled region selection)
try:
    region_map = np.load("data/region_map.npy")
except FileNotFoundError:
    region_map = np.zeros((int(w / SETTINGS["map_scale"]), int(h / SETTINGS["map_scale"])))

# Initialize counters and flags
clock = pygame.time.Clock()
ticks = 0
frame_count = 0  # We'll increment this every frame
# Instead of reading from the folder, we reset the counter to 0 at startup.
num_images = 0

show_stats = False
show_webcam = False
selection_screen = True
calibrate_screen = False
calibrate_idx = 0
collect_screen = False
collect_state = 0  # 0: starting region; 1: moving & saving; 2: done
collect_mode_start_time = 0
collect_start_region = get_undersampled_region(region_map, SETTINGS["map_scale"])
new_target = None  # For collection mode target

def save_data(l_eye, r_eye, face, face_align, head_pos, angle, targetx, targety):
    global num_images
    data_id = num_images + 1
    for (path, img) in zip(data_dirs, (l_eye, r_eye, face, face_align, head_pos)):
        cv2.imwrite(f"{path}/{data_id}.jpg", img)
    csv_writer.writerow([data_id, targetx, targety, angle])
    region_map[int(targetx / SETTINGS["map_scale"]), int(targety / SETTINGS["map_scale"])] += 1
    num_images += 1

def cleanup():
    np.save("data/region_map.npy", region_map)
    plot_region_map("data/region_map.png", region_map, SETTINGS["map_scale"], cmap="inferno")
    data_file.close()
    detector.close()
    pygame.quit()
    sys.exit(0)

while True:
    # Increment frame counter each iteration.
    frame_count += 1

    # Process events once per loop
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.VIDEORESIZE:
            w, h = pygame.display.get_surface().get_size()
            calibration_zones = get_calibration_zones(w, h, SETTINGS["target_radius"])
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                cleanup()
            elif event.key == K_c:
                show_webcam = not show_webcam
            elif event.key == K_s:
                show_stats = not show_stats
            elif selection_screen:
                if event.key == K_1:
                    selection_screen = False
                    calibrate_screen = True
                    calibrate_idx = 0
                    target.moving = False
                    target.color = COLOURS["blue"]
                elif event.key == K_2:
                    selection_screen = False
                    collect_screen = True
                    target.color = COLOURS["green"]
                    collect_mode_start_time = pygame.time.get_ticks()
                    collect_state = 0
            elif calibrate_screen:
                if event.key == K_SPACE:
                    # Save data for current dot and move to next dot.
                    if calibrate_idx < len(calibration_zones):
                        save_data(l_eye, r_eye, face, face_align, head_pos, angle, target.x, target.y)
                    calibrate_idx += 1
            elif collect_screen:
                if event.key == K_SPACE:
                    # Optionally, SPACE can force a state change in collect mode.
                    collect_state = 2

    # Update background color slightly (for corneal reflection variation)
    bg_origin = screen.get_at((0, 0))
    inc = 1 if bg_origin[0] <= COLOURS["black"][0] else -1
    bg = (bg_origin[0] + inc, bg_origin[1] + inc, bg_origin[2] + inc, bg_origin[3])
    screen.fill(bg)

    # Get frame from detector; if detection fails, skip this frame.
    result = detector.get_frame()
    if result is None or any(v is None for v in result):
        pygame.display.update()
        continue
    l_eye, r_eye, face, face_align, head_pos, angle = result

    # Optionally display webcam view
    if show_webcam:
        try:
            combined = np.rot90(
                np.vstack((
                    np.hstack((head_pos, face_align)),
                    np.hstack((l_eye, r_eye))
                )),
                1,
            )
            pygame.surfarray.blit_array(webcam_surface, bgr_to_rgb(combined))
            screen.blit(webcam_surface, (0, SETTINGS["target_radius"] * 2))
            angle_text = font_small.render(f"{round(angle, 1)} deg", True, COLOURS["green"])
            fps_text = font_small.render(f"{int(clock.get_fps())} fps", True, COLOURS["green"])
            webcam_surface.blit(angle_text, (2, 0))
            webcam_surface.blit(fps_text, (2, webcam_surface.get_height() - fps_text.get_height()))
        except Exception as e:
            print("Webcam view error:", e)

    # Optionally display region map stats
    if show_stats:
        if region_map.max() > 0:
            region_map_3d = region_map * (255.0 / region_map.max())
        else:
            region_map_3d = region_map
        region_map_3d = np.broadcast_to(region_map_3d[..., None], region_map_3d.shape + (3,))
        region_surface = pygame.surfarray.make_surface(region_map_3d)
        region_surface = pygame.transform.scale(region_surface, (w, h))
        screen.blit(region_surface, (0, 0))
        num_text = font_normal.render(f"# of images: {num_images}", True, COLOURS["white"])
        dim_text = font_normal.render(f"Image dims: {SETTINGS['image_size']}x{SETTINGS['image_size']}", True, COLOURS["white"])
        cov_text = font_normal.render(f"Coverage: {round(np.count_nonzero(region_map > 0) / region_map.size * 100, 2)}%", True, COLOURS["white"])
        th = cov_text.get_height()
        screen.blit(cov_text, (10, h - th * 4))
        screen.blit(num_text, (10, h - th * 3))
        screen.blit(dim_text, (10, h - th * 2))

    # -------------------------
    # MODE: SELECTION SCREEN
    # -------------------------
    if selection_screen:
        sel_text1 = font_normal.render("(1) Calibrate | (2) Collect", True, COLOURS["white"])
        sel_text2 = font_normal.render("(C) Toggle cam | (S) Show stats | (ESC) Quit", True, COLOURS["white"])
        screen.blit(sel_text1, (10, h * 0.3))
        screen.blit(sel_text2, (10, h * 0.6))

    # -------------------------
    # MODE: CALIBRATION
    # -------------------------
    elif calibrate_screen:
        if calibrate_idx < len(calibration_zones):
            target.x, target.y = calibration_zones[calibrate_idx]
            target.render(screen)
        else:
            screen.fill(COLOURS["black"])
            done_text = font_normal.render("Done", True, COLOURS["white"])
            screen.blit(done_text, done_text.get_rect(center=screen.get_rect().center))
            pygame.time.delay(1000)
            calibrate_idx = 0
            selection_screen = True
            calibrate_screen = False

    # -------------------------
    # MODE: DATA COLLECTION
    # -------------------------
    elif collect_screen:
        if collect_state == 0:
            # Show starting region for 2 seconds
            target.x, target.y = collect_start_region
            target.render(screen)
            if pygame.time.get_ticks() - collect_mode_start_time > 2000:
                collect_state = 1
        elif collect_state == 1:
            # Compute a new target every frame.
            if SETTINGS.get("only_edges", False):
                new_x = random.choice([0, w])
                new_y = random.choice([0, h])
                new_target = (new_x, new_y)
            elif SETTINGS.get("focus_edges", False):
                new_x = (beta.rvs(SETTINGS["beta_a"], SETTINGS["beta_b"], size=1) * w)[0]
                new_y = (beta.rvs(SETTINGS["beta_a"], SETTINGS["beta_b"], size=1) * h)[0]
                new_target = (new_x, new_y)
            else:
                new_target = get_undersampled_region(region_map, SETTINGS["map_scale"])
            # Save data every few frames, according to skip_frames.
            if frame_count % SETTINGS["skip_frames"] == 0:
                save_data(l_eye, r_eye, face, face_align, head_pos, angle, target.x, target.y)
            target.move(new_target, ticks)
            target.render(screen)
        elif collect_state >= 2:
            screen.fill(COLOURS["black"])
            done_text = font_normal.render("Done", True, COLOURS["white"])
            screen.blit(done_text, done_text.get_rect(center=screen.get_rect().center))
            pygame.time.delay(1000)
            collect_state = 0
            selection_screen = True
            collect_screen = False

    ticks = clock.tick(SETTINGS["record_frame_rate"])
    pygame.display.update()
