import cv2
import time
import threading
from flask import Response, Flask

# Image frame sent to the Flask object
global webcam_video_frame
webcam_video_frame = None

# Use locks for thread-safe viewing of frames in multiple browsers
global webcam_thread_lock 
webcam_thread_lock = threading.Lock()

global webcam_robot

# GStreamer Pipeline to access the Raspberry Pi camera
# GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'
# STREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=5/1 ! nvvidconv flip-method=0 ! video/x-raw, width=410, height=308, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'
# if choose a divide 960,616 by 8 or 16 then output is wrong.
WEBCAM_GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=328, height=246, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'

# Create the Flask object for the application
webcam_app = Flask(__name__)

def webcam_captureFrames():
    global webcam_video_frame, webcam_thread_lock

    # Video capturing from OpenCV
    video_capture = cv2.VideoCapture(WEBCAM_GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)

    while True and video_capture.isOpened():
        return_key, frame = video_capture.read()
        if not return_key:
            break

        # Create a copy of the frame and store it in the global variable,
        # with thread safe access
        with webcam_thread_lock:
            webcam_video_frame = frame.copy()

        # if appropriate, store image in designated robot training directory 
        image_loc = webcam_robot.capture_frame_location()
        if image_loc != None:
            return_key, encoded_image = cv2.imencode(".jpg", webcam_video_frame)
            if return_key:
                with open(image_loc, 'wb') as f:
                    f.write(encoded_image)
            # free the robot to move;
            # if bottleneck, optimistically move up before encode/write
            webcam_robot.capture_frame_completed()

        # if appropriate, return frame
        if webcam_robot.get_NN_mode() and webcam_robot.ready_for_capture():
            webcam_robot.capture_frame(webcam_video_frame)
            # return_key, encoded_image = cv2.imencode(".jpg", webcam_video_frame)
            # if return_key:
            #   webcam_robot.capture_frame(encoded_image)
        
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

    video_capture.release()

def endpoint():
    results_bulk_iter = iter([])
    response = Response(stream_with_context(webcam_bulk_update_streamed_response(results_bulk_iter)),
                        mimetype='text/plain')
    return response

def webcam_bulk_update_streamed_response(results):
    for updated, _ in results:
        yield str(counter).decode('utf-8')
        
def webcam_encodeFrame():
    global webcam_thread_lock
    while True:
        # Acquire thread_lock to access the global video_frame object
        with webcam_thread_lock:
            global webcam_video_frame
            if webcam_video_frame is None:
                continue
            return_key, encoded_image = cv2.imencode(".jpg", webcam_video_frame)
            if not return_key:
                continue

        # Output image as a byte array
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encoded_image) + b'\r\n')

@webcam_app.route("/")
def streamFrames():
    return Response(webcam_encodeFrame(), mimetype = "multipart/x-mixed-replace; boundary=frame")

# Must be the last call in the main thread of execution
def webcam_run(robot):
    global webcam_robot

    webcam_robot = robot
    # Create a thread and attach the method that captures the image frames, to it
    process_thread = threading.Thread(target=webcam_captureFrames)
    process_thread.daemon = True

    # Start the thread
    process_thread.start()

    # start the Flask Web Application
    # While it can be run on any feasible IP, IP = 0.0.0.0 renders the web app on
    # the host machine's localhost and is discoverable by other machines on the same network 
    # app.run("0.0.0.0", port=8000)
    webcam_app.run("10.0.0.31", port=8080)

