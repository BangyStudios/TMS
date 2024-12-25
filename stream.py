import cv2
import rtsp

USERNAME = "admin"
PASSWORD = "123456"

with rtsp.Client(rtsp_server_uri=f"rtsp://192.168.1.6:554/chID=13&streamType=main&linkType=tcp") as stream:
    stream.preview()