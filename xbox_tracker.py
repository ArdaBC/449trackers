import time
import pygame
import threading
import cv2
import dlib
from collections import deque
from scipy.spatial import distance as dist

# Configuration
auto_log_name = time.strftime("%H-%M_%d-%m", time.localtime())
LOG_FILE = auto_log_name + "_controller_log.txt"

EAR_THRESHOLD = 0.25
SMOOTHING_WINDOW = 5
MIN_CLOSE_DURATION = 0.10

blink_state = 0
close_start_time = None
blink_count = 0
ear_buffer = deque(maxlen=SMOOTHING_WINDOW)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

controller_map = {
    0: "A",
    1: "B",
    2: "X",
    3: "Y",
    8: "LB",
    9: "RB",
    10: "LT",
    11: "RT",
    12: "L3",  # Left Stick Press
    13: "R3",  # Right Stick Press
}

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def track_actions():
    global blink_state, close_start_time, blink_count

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No controllers connected.")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    cap = cv2.VideoCapture(0)

    with open(LOG_FILE, "a") as log_file:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            # Blink detection
            for face in faces:
                landmarks = predictor(gray, face)
                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                ear_buffer.append(avg_ear)
                smoothed_ear = sum(ear_buffer) / len(ear_buffer)

                if smoothed_ear < EAR_THRESHOLD and blink_state == 0:
                    blink_state = 1
                    close_start_time = time.time()
                elif smoothed_ear > EAR_THRESHOLD and blink_state == 1:
                    blink_state = 0
                    if close_start_time is not None:
                        close_duration = time.time() - close_start_time
                        if close_duration >= MIN_CLOSE_DURATION:
                            blink_count += 1
                    close_start_time = None

                # Visualizations
                for (x, y) in left_eye + right_eye:
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                cv2.putText(frame, f"Blink Count: {blink_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # -----------------------------------
            # Poll joystick events for each loop
            # -----------------------------------
            pygame.event.pump()

            # 1) Get the system mouse position (if you still want to track it)
            mouse_coords = pygame.mouse.get_pos()

            # 2) Get L3/R3 axis values (this is new!)
            l3_x = joystick.get_axis(0)  # Typically left stick horizontal
            l3_y = joystick.get_axis(1)  # Typically left stick vertical
            r3_x = joystick.get_axis(4)  # Typically right stick horizontal
            r3_y = joystick.get_axis(5)  # Typically right stick vertical

            # 3) Decide how you want to handle L3 movement
            #    - For now, let's just log it as well.
            #    - If you want to *control the OS mouse*, you'd update the mouse coords and call pygame.mouse.set_pos().
            #    - Or do something else game-specific with it.
            
            # Track which button is being pressed
            action = "None"
            for button_id in controller_map:
                if joystick.get_button(button_id):
                    action = controller_map[button_id]
                    break

            # Prepare log output
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_text = (
                f"{timestamp} - "
                f"({l3_x:.2f}, {l3_y:.2f}) - "
                f"({r3_x:.2f}, {r3_y:.2f}) - "
                f"{action} - {blink_count}\n"
            )

            # Write to log
            log_file.write(log_text)
            log_file.flush()

            # Display the frame
            cv2.imshow("Blink Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Slight sleep
            time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()

def start_logging():
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
