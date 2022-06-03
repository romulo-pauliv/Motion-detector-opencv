# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                              AttendanceProject.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                             https://(address.com) ||
# |                                                                                                      Version 1.0  ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
import cv2
import winsound
import colorama as clrm
from datetime import datetime
import keyboard
# var -----------------------------------------------------------------------------------------------------------------|
csvLog = True               # Generate log in .csv file
saveImg = True              # Save detection images
sound = True                # Sound when detected
printImg = True             # Show your camera
contourArea = 5000          # Number of pixels needed to generate a detection
directory = 0               # Camera Directory. 0 to use your main webcam

# DirCsv: .csv file directory. You need to create a csv file called "motion-detector.csv" and paste its directory
# into the variable. # Ex.: dirCsv = r"C:\Users\computer\Documents\motion-detector.csv"
dirCsv = r"motion-detector.csv"

# DirImg: Directory of the folder where the images will be saved. It is necessary to create a folder called -
# "MotionDetectionImage" and paste its directory in the variable.
# Ex.: dirImg = r"C:\Users\computer\Documents\MotionDetectionImage"
dirImg = r'MotionDetectionImage'


cam = cv2.VideoCapture(directory, cv2.CAP_DSHOW)  # Video file reading
dtString = None                                   # Global variable
# function ------------------------------------------------------------------------------------------------------------|


def log(lognum):
    if lognum == 0:
        print(clrm.Fore.YELLOW +
              "|-------------------- Motion Detector --------------------|" +
              clrm.Style.RESET_ALL)

    if lognum == 1:
        print(clrm.Fore.WHITE + "Video directory/camera |-----------| ", end="")

    if lognum == 2:
        print(clrm.Fore.WHITE + "Generate log in .csv   |-----------|", end="")

    if lognum == 3:
        print(clrm.Fore.WHITE + "Save images            |-----------|", end="")
    if lognum == 4:
        print(clrm.Fore.YELLOW +
              "|------------------------>> Initializing directory capture|" +
              clrm.Style.RESET_ALL)
    if lognum == 5:
        print(clrm.Fore.WHITE + "Detection sound        |-----------|", end="")
    if lognum == 6:
        print(clrm.Fore.WHITE + "Print Video            |-----------|", end="")


def logBlock():
    log(0), log(1), print(clrm.Fore.LIGHTMAGENTA_EX + str(directory) + clrm.Style.RESET_ALL)
    log(2), print(clrm.Fore.GREEN if csvLog else clrm.Fore.RED, str(csvLog))
    log(3), print(clrm.Fore.GREEN if saveImg else clrm.Fore.RED, str(saveImg))
    log(5), print(clrm.Fore.GREEN if sound else clrm.Fore.RED, str(sound))
    log(6), print(clrm.Fore.GREEN if printImg else clrm.Fore.RED, str(printImg))
    log(4)


def csvMotionDetection():
    with open(dirCsv, 'r+') as f:
        myDataList = f.readline()
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        now = datetime.now()
        dt = now.strftime('%d/%M/%Y-%H:%M:%S')
        f.writelines(f'\nMotion Detection, {dt}')
# code ----------------------------------------------------------------------------------------------------------------|


logBlock()
while cam.isOpened():
    # The method/function combines VideoCapture::grab() and VideoCapture::retrieve() in one call. This is the
    # most convenient method for reading video files or capturing data from decode and returns the just
    # grabbed frame. If no frames has been grabbed (camera has been disconnected, or there are no more
    # frames in video file), the method returns false and the function returns empty image
    # (with %cv::Mat, test it with Mat::empty()).
    _, frame1 = cam.read()
    _, frame2 = cam.read()

    # Calculates the per-element absolute difference between two arrays or between an array and a scalar.
    diff = cv2.absdiff(src1=frame1, src2=frame2)

    # Convert np.array to grayscale.
    gray = cv2.cvtColor(src=diff, code=cv2.COLOR_RGB2GRAY)

    # The function convolve the source image with the specified Gaussian kernel. In-place filtering is supported.
    blur = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)

    # The function applies fixed-level thresholding to a multiple-channel array. The function is typically
    # ed to get a bi-level (binary) image out of a grayscale image ( #compare could be also used for
    # this purpose) or for removing a noise, that is, filtering out pixels with too small or too large
    # values. There are several types of thresholding supported by the function. They are determined by
    # type parameter.
    _, thresh = cv2.threshold(src=blur, thresh=20, maxval=255, type=cv2.THRESH_BINARY)

    # The function dilates the source image using the specified structuring element that determines the
    # shape of a pixel neighborhood over which the maximum is taken:
    # \f[\texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\f]
    dilated = cv2.dilate(src=thresh, kernel=None, iterations=3)

    # The function retrieves contours from the binary image using the algorithm @cite Suzuki85 . The contours
    # are a useful tool for shape analysis and object detection and recognition.
    contours, _ = cv2.findContours(image=dilated, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    for c in contours:  # iterable object with contour coordinates
        if cv2.contourArea(c) < contourArea:
            # cv2.contourArea
            # The function computes a contour area. Similarly to moments, the area is computed using the Green
            # formula. Thus, the returned area and the number of non-zero pixels, if you draw the contour using
            # #drawContours or #fillPoly , can be different. Also, the function will most certainly give a wrong
            # results for contours with self-intersections.
            continue

        # @brief Calculates the up-right bounding rectangle of a point set or non-zero pixels of gray-scale image.
        # The function calculates and returns the minimal up-right bounding rectangle for the specified point set or
        # non-zero pixels of gray-scale image.
        x, y, w, h = cv2.boundingRect(c)

        # Log to .csv file
        csvMotionDetection() if csvLog else csvLog

        # The function cv::rectangle draws a rectangle outline or a filled rectangle whose two opposite corners
        cv2.rectangle(img=frame1, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=1)

        # The function cv::putText renders the specified text string in the image. Symbols that cannot be rendered
        # using the specified font are replaced by question marks.
        cv2.putText(img=frame1, text=str(datetime.now()), org=(20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 255, 255), thickness=1)

        dtString = datetime.now()                           # Data Time
        logDate = dtString.strftime('%d%M%Y_%H%M%S')        # Format datatime str

        #  The function imwrite saves the image to the specified file. The image format is chosen based on the
        #  filename extension (see cv::imread for the list of extensions).
        cv2.imwrite(dirImg + "\\" + str(logDate) + '.png', frame1) if saveImg else saveImg

        # Warning sound
        winsound.Beep(4000, 10) if sound else sound

        # Log console
        print(clrm.Fore.LIGHTMAGENTA_EX + 'Motion Detection - ' +
              clrm.Fore.WHITE + str(dtString) + clrm.Style.RESET_ALL)
        print('.csv log |------------|', end='')
        print((clrm.Fore.GREEN if csvLog else clrm.Fore.RED) + str(csvLog) + clrm.Style.RESET_ALL)
        print('save img |------------|', end='')
        print((clrm.Fore.GREEN if saveImg else clrm.Fore.RED) + str(saveImg) + clrm.Style.RESET_ALL)

    # The function waitKey waits for a key event infinitely (when \f$\texttt{delay}\leq 0\f$ ) or for delay
    # milliseconds, when it is positive. Since the OS has a minimum time between switching threads, the
    # function will not wait exactly delay ms, it will wait at least delay ms, depending on what else is
    # running on your computer at that time. It returns the code of the pressed key or -1 if no key was
    # pressed before the specified time had elapsed. To check for a key press but not wait for it, use
    # #pollKey.
    if cv2.waitKey(5) == ord('q'):
        break

    # function stops code when clicked 'alt+ctrl'
    if keyboard.is_pressed('alt + ctrl'):
        break

    # The function imshow displays an image in the specified window. If the window was created with the
    # cv::WINDOW_AUTOSIZE flag, the image is shown with its original size, however it is still limited
    # by the screen resolution. Otherwise, the image is scaled to fit the window.
    cv2.imshow("Motion Detection", frame1) if printImg else printImg

# The method is automatically called by subsequent VideoCapture::open and by VideoCapture destructor.
cam.release()

# The function destroyAllWindows destroys all the opened HighGUI windows
cv2.destroyAllWindows()
