import time
import win32api
import win32con
import threading
import cv2
import dlib
from collections import deque
from scipy.spatial import distance as dist

# ----------------------
# Configuration
# ----------------------
auto_log_name = time.strftime("%H-%M_%d-%m", time.localtime())
LOG_FILE = auto_log_name + "_log.txt"

# Blink detection parameters
EAR_THRESHOLD = 0.25         # Eye Aspect Ratio threshold
SMOOTHING_WINDOW = 5         # Number of frames used for moving-average
MIN_CLOSE_DURATION = 0.10    # Only count blink if eyes stayed closed at least X seconds
                            # Set to 0 to disable this feature.

# We'll track the "state" of the eyes:
#   0 = open
#   1 = closed
blink_state = 0
close_start_time = None
blink_count = 0

# Rolling buffer for smoothing
ear_buffer = deque(maxlen=SMOOTHING_WINDOW)

# Face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Simple key map for logging
key_map = {
    0x57: "W",
    0x41: "A",
    0x53: "S",
    0x44: "D",
    0x52: "R",
    0x43: "C",
    win32con.VK_SHIFT: "Shift",
    win32con.VK_CONTROL: "Ctrl",
    win32con.VK_MENU: "Alt",
    win32con.VK_SPACE: "Space",
    win32con.VK_ESCAPE: "Esc",
    0x31: "1",
    0x32: "2",
    0x33: "3",
    0x34: "4",
}

def calculate_ear(eye):
    """
    Compute the Eye Aspect Ratio (EAR) for a given eye.
    eye: list of (x, y) points corresponding to the eye landmarks
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def track_actions():
    global blink_state, close_start_time, blink_count

    cap = cv2.VideoCapture(0)

    with open(LOG_FILE, "a") as log_file:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            # ----------------------------------------------------------
            # For each detected face, compute EAR and update blink state
            # ----------------------------------------------------------
            for face in faces:
                landmarks = predictor(gray, face)

                # Extract eye coordinates
                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

                # Calculate EAR for both eyes, then average
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                # Smooth the EAR using a rolling average
                ear_buffer.append(avg_ear)
                smoothed_ear = sum(ear_buffer) / len(ear_buffer)

                # -----------------------
                # Rise-and-Fall Logic
                # -----------------------
                if smoothed_ear < EAR_THRESHOLD and blink_state == 0:
                    # Eyes just closed
                    blink_state = 1
                    close_start_time = time.time()

                elif smoothed_ear > EAR_THRESHOLD and blink_state == 1:
                    # Eyes just opened
                    blink_state = 0
                    if close_start_time is not None:
                        close_duration = time.time() - close_start_time
                        # Only count it as a blink if it stayed closed long enough
                        if close_duration >= MIN_CLOSE_DURATION:
                            blink_count += 1
                    close_start_time = None

                # Visualize the eyes and blink count
                for (x, y) in left_eye + right_eye:
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                cv2.putText(frame, f"Blink Count: {blink_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --------------
            # Log actions
            # --------------
            action = "None"
            coords = win32api.GetCursorPos()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # Mouse clicks
            if win32api.GetKeyState(win32con.VK_LBUTTON) < 0:
                action = "Left Click"
            elif win32api.GetKeyState(win32con.VK_RBUTTON) < 0:
                action = "Right Click"
            else:
                # Keyboard actions
                for vk, name in key_map.items():
                    if win32api.GetKeyState(vk) < 0:
                        action = name

            # Write to log
            log_file.write(f"{timestamp} - {coords} - {action} - {blink_count}\n")
            log_file.flush()

            # Show the frame
            cv2.imshow("Blink Detection", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Slightly reduce sleep for better frame rate, so we don't miss quick blinks
            time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()

def start_logging():
    """
    Runs track_actions() in a background thread, 
    while this main thread just sleeps indefinitely.
    """
    thread = threading.Thread(target=track_actions)
    thread.daemon = True
    thread.start()

    try:
        while True:
            time.sleep(1)
    except:
        pass

if __name__ == "__main__":
    start_logging()
