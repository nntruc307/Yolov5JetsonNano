import sys
import cv2 
import imutils
import time
from yolov5Det import YoloV5TRT
import serial
import Jetson.GPIO as GPIO
import threading
import numpy as np

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1920x1080 @ 30fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen
# Notice that we drop frames if we fall outside the processing time in the appsink element
def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=2,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
def remove_duplicates_preserve_order(lst): # Function to remove duplicate, then save the memory
    seen = {}
    new_lst = []
    for i, item in enumerate(lst):
        if item not in seen:
            seen[item] = i
            new_lst.append(item)
        else:
            seen[item] = i
    new_lst = [k for k, v in sorted(seen.items(), key=lambda x: x[1])]
    return new_lst

def sign_catch(list_det, thresh): # List of detected signs, thresh is the minimum number of detected signs have presented, to avoid noise and wrong detections
    global max_speed_limited
    global min_speed_limited
    priority_list = ['SL 120','SL 100','SL 80', 
                    'SL 70','SL 60','SL 50',
                    'Start Res', 'End Res', 'End SL', 'SM 60']

    sign_dict = {'SL 50': 0, 'SL 60': 0, 'SL 70': 0,
                'SL 80': 0, 'SL 100': 0, 'SL 120': 0,
                'End Res': 0, 'End SL': 0, 'Start Res': 0, 'SM 60': 0}

    for i in list_det:
        for key in sign_dict:
            if i == key:
                sign_dict[key] +=1
    filtered_dict = {k: thresh for k, v in sign_dict.items() if v > thresh and k in priority_list}
    if not filtered_dict:
        return ""
    detected_sign = sorted(filtered_dict, key=lambda x: priority_list.index(x)) # Sort the speed limit signs following priority list

    if "SL " in detected_sign[0]: # Maximum speed limit in case detected Speed Limit Signs
        if "SL 120" in detected_sign[0] or "SL 100" in detected_sign[0]:
            max_speed_limited.append(int(detected_sign[0].replace("SL ","")))
            min_speed_limited.append(60)
        else:
            max_speed_limited.append(int(detected_sign[0].replace("SL ","")))
            min_speed_limited.append(0)

    elif "Start Res" in detected_sign[0]: # Maximum speed limit in case detected Start of Residential Area
        max_speed_limited.append(50)
        min_speed_limited.append(0)
    elif "End Res" in detected_sign[0]: # Maximum speed limit in case detected End of Residential Area
        max_speed_limited.append(70)
        min_speed_limited.append(0)
    elif "End SL" in detected_sign[0]:
        max_speed_limited.append(max_speed_limited[-1] + 10)
        min_speed_limited.append(0)

    max_speed_limited = remove_duplicates_preserve_order(max_speed_limited)
    min_speed_limited = remove_duplicates_preserve_order(min_speed_limited)

# Read Vehicle Speed from CAN function ===================================================================
def UART_Read():
    # threading.Timer(timer_interval, UART_Read).start()
    global veh_spd_fr_can
    try:
        serial_port = serial.Serial(
            port="/dev/ttyTHS1",
            baudrate=115200,
            timeout=0.01,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        # Wait a second to let the port initialize
        #time.sleep(1)
        serial_port.write("NVIDIA Jetson Nano Developer Kit\r\n".encode())

        data = serial_port.read(8)
        veh_spd_raw = data.decode("utf-8", errors="ignore").strip()
        # Add vehicle speed from CAN to buffer
        if veh_spd_raw is not "":
            veh_spd = veh_spd_raw.replace("v: ","")
            try:
                veh_spd = float(veh_spd)
                veh_spd_fr_can.append(veh_spd)
            except TypeError as e:
                print("Error to convert float: {}".format(str(e)))
                pass
        veh_spd_fr_can = remove_duplicates_preserve_order(veh_spd_fr_can)
        # Close the serial connection
        serial_port.close()
    
    except Exception as exception_error:
        print("Error occurred. Exiting Program")
        print("Error: " + str(exception_error))
        pass

# Control LED function ======================================================================
def control_led(mode):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(33, GPIO.OUT) # Setup for pin 33
    
    if mode == "off":
        GPIO.output(33, GPIO.LOW)
    
    # elif mode == "slow_blink":
    #     GPIO.output(33, GPIO.HIGH)
    #     time.sleep(0.5)
    #     GPIO.output(33, GPIO.LOW)
    #     time.sleep(0.5)
    
    elif mode == "on":
        GPIO.output(33, GPIO.HIGH)
    
    GPIO.cleanup()


# Global variable define================================
timer_interval = 1.0  # Interval in seconds
veh_spd_fr_can = [0] # Buffer to storage vehicle speed from CAN
fps_start_time = time.time() # Pre-define start time counting FPS
fps_counter = 0 # Pre-define FPS counting vairable
gap_time_count = 10 # Define gap time between each traffic signs were detected
sign_start_time = time.time() 
speed_limit_signs = [] # Buffer to storage traffic signs were detected. Will be reset if in gap time it does not detect any signs
max_speed_limited = [50] # Buffer to storage maximum speed, first define 50kmh
min_speed_limited = [0] # Buffer to storage minimum speed, first define 0kmh
count_threh = 4 # Variable to define number of detect to recognize
# Global variable define================================

model = YoloV5TRT(library = "yolov5/build/libmyplugins.so", engine = "TSD_6_640.engine", conf = 0.7)
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER) # Stream video from CSI camera
# cap = cv2.VideoCapture("TSD_Video_2.mp4") # Stream video from source video
window_title = "Traffic Signs Detection" # Set window title
window_width = 1440 # Display window with specific Width
window_height = 810 # Display window with specific height
frame_width = 960 # 

while True:
    ret, frame = cap.read()
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, window_width, window_height)
    frame = imutils.resize(frame, width=frame_width)
    detections, t = model.Inference(frame)

    for obj in detections:
        print(obj['class'], obj['conf'], obj['box'])
        speed_limit_signs.append(obj['class'])
        sign_start_time = time.time()
    # print(sign_start_time)
    if len(speed_limit_signs) > 0:
        if (time.time() - sign_start_time) > gap_time_count: # Reset the buffer if during gap time not detect any signs
            speed_limit_signs = []
            
    x, y = 0, window_height - 370
    width, height = window_width, 80

    alpha = 0.6  # Transparency value (0.0 - fully transparent, 1.0 - fully opaque)

    # Create a copy of the image
    overlay = frame.copy()

    # Draw a filled rectangle on the overlay image
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (250, 250, 250), -1)

    # Apply the overlay onto the original image
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Get speed limit
    sign_catch(speed_limit_signs, thresh=count_threh)
    # print(max_speed_limited) # Print max speed limit buffer
    max_spd_text = "Max Speed: " + str(max_speed_limited[-1]) + " km/h"
    cv2.putText(frame, max_spd_text, (x+450, window_height - 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    # print(max_spd_text)
    min_spd_text = "Min Speed: " + str(min_speed_limited[-1]) + " km/h"
    speed_limit_text = "Speed Limit: Min : " + str(min_speed_limited[-1]) + " km/h" 
    cv2.putText(frame, min_spd_text, (x+10, window_height - 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # print(min_spd_text)

    # Read current vehicle speed from CAN
    UART_Read()
    # print(veh_spd_fr_can) # Print vehicle speed from CAN buffer
    current_veh_spd = veh_spd_fr_can[-1]
    veh_spd_disp = "Current Speed: {:.0f} km/h".format(current_veh_spd)
    # print(veh_spd_disp)
    # Put current vehicle speed to display window
    

    
    # LED control
    if current_veh_spd >= (float(max_speed_limited[-1]) + 5):
        control_led("on")
        cv2.putText(frame, veh_spd_disp + " (Speed Too High!!!)", (x+10, window_height - 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif (float(max_speed_limited[-1])) < current_veh_spd < (float(max_speed_limited[-1]) + 5):
        control_led("on")
        cv2.putText(frame, veh_spd_disp + " (Speed High)", (x+10, window_height - 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    elif min_speed_limited[-1] == 60:
        if current_veh_spd <= (float(min_speed_limited[-1]) - 5):
            control_led("on")
            cv2.putText(frame, veh_spd_disp + " (Speed Too Low!!!)", (x+10, window_height - 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif (float(min_speed_limited[-1]) - 5) < current_veh_spd < (float(min_speed_limited[-1])):
            control_led("on")
            cv2.putText(frame, veh_spd_disp + " (Speed Low)", (x+10, window_height - 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
        else:
            control_led("off")
            cv2.putText(frame, veh_spd_disp, (x+10, window_height - 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)
    else:
        control_led("off")
        cv2.putText(frame, veh_spd_disp, (x+10, window_height - 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)

    #Calculate FPS
    fps_counter += 1
    if (time.time() - fps_start_time) > 1:
        fps = fps_counter / (time.time() - fps_start_time)
        fps_text = "FPS:  {:.1f}".format(fps)
        # print(fps_text)
        fps_counter = 0
        fps_start_time = time.time()
    # print("FPS per image: {:.1f}".format(1/t))
    # Put FPS text to the frame
    if 'fps_text' in locals():
        cv2.putText(frame, fps_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show window
    cv2.imshow(window_title, frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()