import cv2

def main():
    """
        Run a test GUI with  webcam capture image
    """

    cap = cv2.VideoCapture(0)

    while(True):
        # Get Webcam
        ret, frame = cap.read()

        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        roi_color = None
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            eye_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex+5,ey+5),(ex+ew-5,ey+eh-5),(0,255,0),2)
                th1 = cv2.adaptiveThreshold(eye_gray[ey+5:ey+eh-5, ex+5:ex+ew-5], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
                th2 = cv2.dilate(th1, None)
                th3 = cv2.dilate(th2, None)
                th4 = cv2.erode(th3, None)
                th5 = cv2.erode(th4, None)
                th6, contours, hierarchy = cv2.findContours(th5, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                rc = cv2.minAreaRect(contours[0])
                box = cv2.boxPoints(rc)
                xs, ys = zip(*box)
                cv2.circle(roi_color[ey+5:ey+eh-5, ex+5:ex+ew-5], (int(sum(xs) / 4), int(sum(ys) / 4)), 2, (0, 0, 200), 2)

                cv2.imshow('Eye', th1)

        # Display the resulting frame for debugging
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
