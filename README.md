# BAD Drone Project

This project aims to develop a drone that can navigate a preprogrammed flight path and detect and avoid obstacles. The current implementation utilizes 2 machine learning models running in parallel: YOLOv5 Nano V6 and MiDaS.

YOLOv5 Nano V6 is a pretrained and lightweight computer vision model that can identify objects quickly from pictures. Its efficiency makes it great for taking in a live feed from the drone and identifying objects in real-time.

MiDaS is another lightweight computer vision model that is pretrained. MiDaS is a monocular depth estimation model that attempts to estimate how far an object is from a single image. If the drone gets too close to an object, it will attempt to take action to avoid the obstacle.

This program also features a live stream view of the drone while it goes through its preprogrammed route and will save the raw video file (without any processing or computer vision artifacts).
