import cv2

URL = "rtsp://tanay:tanay123@192.168.1.129:554/cam/realmonitor?channel=1&subtype=0"

source = """rtspsrc location={} latency=0 ! queue ! rtph264depay
    ! h264parse ! avdec_h264 ! videoconvert ! appsink""".format(URL)

print(source)
cap = cv2.VideoCapture(URL)

if not cap.isOpened():
    print("Cannot capture from camera. Exiting.")
    quit()


while True:

    ret, frame = cap.read()
    #
    if ret == False:
        break


    cv2.imshow("FrameREAD",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break