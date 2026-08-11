import cv2
import os
import time
import threading
import queue
from datetime import datetime

# --- VIDEO WRITER THREAD CLASS ---
class VideoWriterThread:
    """Handles disk writing in a background thread to prevent bottlenecking the camera feed."""
    def __init__(self, filename, fourcc, fps, frame_size):
        self.out = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self.write_frames)
        self.thread.daemon = True 
        self.thread.start()

    def write_frames(self):
        # Keep running until explicitly stopped AND the queue is completely empty
        while self.running or not self.queue.empty():
            try:
                # Pulls a frame from the queue; times out after 0.1s to check 'self.running' again
                frame = self.queue.get(timeout=0.1)
                self.out.write(frame)
            except queue.Empty:
                pass

    def write(self, frame):
        self.queue.put(frame)

    def stop(self):
        self.running = False
        self.thread.join() # Waits for the thread to finish writing remaining frames
        self.out.release()


# --- MAIN SCRIPT ---
# Create the folder for the videos
os.makedirs("typing_videos", exist_ok=True)

# Open the scrcpy v4l2 sink
cap = cv2.VideoCapture(10)

# Attempt to force the capture source to 30 FPS
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open /dev/video10. Is scrcpy running?")
    exit()

# Get the frame width and height to correctly initialize the VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30.0

print("Press SPACE to start/stop recording. Press ESC to quit.")

# State variables for recording
recording = False
writer = None
start_time = 0

# Window Management
window_name = "Video Capture - Press SPACE to record"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
cv2.moveWindow(window_name, 100, 100)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Create a copy of the frame for the UI overlay 
    display_frame = frame.copy()

    if recording:
        # 1. Send the clean frame to the background thread's queue (Instantly returns!)
        writer.write(frame)
        
        # 2. Calculate the elapsed time
        elapsed_time = time.time() - start_time
        mins = int(elapsed_time // 30)
        secs = int(elapsed_time % 30)
        timer_text = f"REC: {mins:02d}:{secs:02d}"
        
        # 3. Add a recording indicator and the timer to the DISPLAY frame only
        cv2.circle(display_frame, (30, 40), 10, (0, 0, 255), -1)
        cv2.putText(display_frame, timer_text, (50, 48), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow(window_name, display_frame)
    
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break
    elif key == 32:  # Spacebar
        if not recording:
            # Start Recording Action
            timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
            filename = f"typing_videos/{timestamp}.mp4"
            
            # Initialize our custom Threaded Video Writer
            writer = VideoWriterThread(filename, fourcc, fps, (frame_width, frame_height))
            recording = True
            start_time = time.time()
            print(f"Started recording: {filename}")
        else:
            # Stop Recording Action
            recording = False
            if writer is not None:
                # This will briefly pause the script while it finishes writing the remaining queued frames
                print("Finishing file write... please wait.")
                writer.stop()
                writer = None
            print("Recording stopped and saved.")

# Cleanup
if writer is not None:
    writer.stop()
cap.release()
cv2.destroyAllWindows()