import time, cv2, copy
import numpy as np
import torch
from threading import Thread
from djitellopy import Tello

recording = False
frame_read = None
drone = None
detection_enabled = False

# Load YOLOv5n6 and MiDaS models
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5n6', pretrained=True) # Used to identify objects
model_depth = torch.hub.load("intel-isl/MiDaS", "MiDaS") # Used to predict distance to objects
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# Initialize depth estimation model
device = torch.device("cuda")
model_depth.to(device)
model_depth.eval()

def main():
    global recording
    global frame_read
    global drone
    global detection_enabled

    drone = Tello()
    drone.connect()

    try:
        recording = True
        drone.streamon()
        frame_read = drone.get_frame_read()
        
        streamer = Thread(target=streamVideo)
        recorder = Thread(target=recordRawVideo)
        streamer.start()
        recorder.start()

        print("The battery is currently at ", drone.get_battery(), "%")
        drone.set_speed(70)

        drone.takeoff()
        detection_enabled = True
        drone.rotate_clockwise(90)
        drone.move_forward(125)
        drone.rotate_clockwise(90)
        drone.move_forward(250)
        drone.rotate_clockwise(90)
        drone.move_forward(125)
        drone.rotate_clockwise(90)
        drone.move_up(75)
        drone.move_forward(250)
        drone.rotate_clockwise(90)
        drone.move_forward(125)
        drone.rotate_clockwise(90)
        drone.move_forward(250)
        drone.rotate_clockwise(90)
        drone.move_forward(125)
        drone.rotate_clockwise(90)
        drone.land()

        recording = False
        recorder.join()
    except Exception as e:
        print('ERROR', e)

        if recording:
            recording = False
            recorder.join()
            print('Ended recording by error')
        if drone.is_flying:
            drone.land()
        
        drone.send_command_without_return('reboot')

def recordRawVideo():
    height, width, _ = frame_read.frame.shape
    video = cv2.VideoWriter(time.strftime("%Y%m%d-%H%M%S") + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while recording:
        if frame_read.frame is not None:
            video.write(frame_read.frame)
        time.sleep(1/30)

    video.release()

def streamVideo():
    while True:
        img = copy.deepcopy(frame_read.frame)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        input_batch = midas_transforms.default_transform(img_rgb)
        
        # Predict depth
        with torch.no_grad():
            input_batch = input_batch.to(device)
            depth = model_depth(input_batch)
            depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False).squeeze().cpu().numpy()

        # Convert depth to a colormap for visualization (optional)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_MAGMA)

        # Run YOLOv5 model
        results = model_yolo(img)

        closeness_threshold = 3750  # Define a threshold for closeness, adjust based on testing

        # Extract bounding boxes and labels for detections
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            label = model_yolo.names[int(cls)]  # Get label name
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            # Calculate the width of the frame and the middle third's boundaries
            frame_width = depth.shape[1]
            middle_third_start = frame_width // 3
            middle_third_end = frame_width * 2 // 3

            # Check if the object's horizontal midpoint is within the middle third of the frame
            object_midpoint = (x1 + x2) // 2
            is_within_middle_third = middle_third_start <= object_midpoint <= middle_third_end

            # Check if the detected object is close
            object_depth = depth[y1:y2, x1:x2]
            if object_depth.size > 0 and np.mean(object_depth) > closeness_threshold - 1000 and is_within_middle_third and detection_enabled:
                cv2.putText(img, "Close Object!", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                if np.mean(object_depth) > closeness_threshold:
                    print('DISTANCE TRIGGERED', np.mean(object_depth))
                    drone.flip_back()
                    drone.move_right(50)
                    drone.land()

        if recording:
            img = cv2.putText(img, 'RECORDING', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Live View', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
