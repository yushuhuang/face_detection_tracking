# Face detection and tracking

OpenCV, face detection, mean shift tracking

The reason I did this project is to combine face detection with tracking.

Sometimes the face detection can not detect faces in frame. For example, some are caused by tilted faces.

This algorithm simply use mean shift tracking to improve the detection consistency.

The consisitency could be useful when trying to keep camera's face detection auto focus smooth.