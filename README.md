# BAD Drone Project

This project aims to develop a drone that can navigate a preprogrammed flight path and can both detect and avoid obstacles. The current implementation utilizes 2 machine learning models running in parallel: YOLOv5 Nano V6 and MiDaS.

YOLOv5 Nano V6 is a pretrained and lightweight computer vision model that can identify objects quickly from pictures. Its efficiency makes it great for taking in a live feed from the drone and identifying objects in real time.

MiDaS is another pretained and lightweight computer vision model. This model attempts to estimate how far an object is from a single image. This model pairs together with YOLOv5 to attempt to calculate distances to different detected objects. If the drone gets too close to an object, it will attempt to avoid the obstacle.