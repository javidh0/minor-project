import cv2
import numpy as np
from scipy.signal import butter
from scipy import signal

MAX_OBJECTS = 1e9
MAX_FRAMES = 1e9

class VideoFeature:
    __videoFileLocation:str = ""
    __maxFrameLength:int = 0
    __getRoiCallback = None
    __colorMatrix:list[np.ndarray] = None

    def __init__(self, videoFileLocation, getRoiCallback, maxFrameLength = 128, fps = 30, maxObjects = 10) -> None:
        self.__videoFileLocation = videoFileLocation
        self.__maxFrameLength = maxFrameLength
        self.__getRoiCallback = getRoiCallback
        self.__maxObjects = maxObjects
        self.__fps = fps
        self.__colorMatrix = []
    
    def getFPS(self):
        return self.__fps

    def setMaxVideoLength(self, maxFrameLength):
        self.__maxFrameLength = maxFrameLength
    
    def readVideo(self):
        cap = cv2.VideoCapture(self.__videoFileLocation)
        self.__colorMatrix.clear()
        frameCount = 0
        objectCount = 0
        b = []
        g = []
        r = []
        y = []

        print("Reading video.. object = -")
        
        while(cap.isOpened() and (objectCount < self.__maxObjects)):
            is_read, frame = cap.read()
            
            if(is_read):
                frame = self.__getRoiCallback(frame)

                b.append(np.mean(frame[:, :, 0]))
                g.append(np.mean(frame[:, :, 1]))
                r.append(np.mean(frame[:, :, 2]))

                ycbcr=cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)                      

                y.append(np.mean(ycbcr[:, :, 0]))

                frameCount += 1

                if(frameCount >= self.__maxFrameLength):
                    self.__colorMatrix.append(np.array([r, g, b, y]))

                    frameCount = 0
                    objectCount += 1
                    r.clear()
                    g.clear()
                    b.clear()
                    y.clear()

                    print(f"Reading video.. object = {objectCount}")

            else:
                print("Video reading Terminated..")
                print("(Object, Frame)", objectCount, frameCount)
                break

        cap.release()
    
    def getColors(self, index):
        if index >= len(self.__colorMatrix):
            return IndexError
        return self.__colorMatrix[index]
    
    def getMaxFrameLength(self):
        return self.__maxFrameLength
    
    def getTotalObjects(self):
        return len(self.__colorMatrix)
    
class ChormFeatures:
    def __init__(self, videoFeature: VideoFeature, order=2):
        self.__videoFeature = videoFeature
        self.__order = order
        self.__fps = videoFeature.getFPS()
        self.__videoReadCounter = 0
        self.__count = 0
        self.__featureImages = []
    
    def setVideoReadCounter(self, videoReadCount):
        self.__videoReadCounter = videoReadCount

    def __ButterBandpass(self, lowcut, highcut):
        nyq = 0.5 * self.__fps
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(self.__order, [low, high], btype='band')
        return b, a

    def __butterBandpassFilter(self, data, lowcut, highcut):
        b, a = self.__ButterBandpass(lowcut, highcut)
        y = signal.filtfilt(b, a, data)
        return y

    def buildCHROM(self):     
        length = self.__videoFeature.getMaxFrameLength()

        r = self.__videoFeature.getColors(self.__count)[0]
        g = self.__videoFeature.getColors(self.__count)[1]
        b = self.__videoFeature.getColors(self.__count)[2]
        y = self.__videoFeature.getColors(self.__count)[3]

        self.__count += 1
        
        r /= np.mean(r)*100
        g /= np.mean(g)*100
        b /= np.mean(b)*100

        r_ = self.__butterBandpassFilter(r, 0.7, 5)
        g_ = self.__butterBandpassFilter(g, 0.7, 5)
        b_ = self.__butterBandpassFilter(b, 0.7, 5)

        self.__X = 3*r_ - 2*g_
        self.__Y = 1.5*r_ + g_ - 1.5*b_
        self.__Y_lum = y

        feature_1 = []
        feature_2 = []
        feature_3 = []

        for i in range(length//2-1):
            feature_1.append(self.__X[i : i+(length//2)+1])
            feature_2.append(self.__Y[i : i+(length//2)+1])
            feature_3.append(self.__Y_lum[i : i+(length//2)+1])
        
        def norm(image):
            return cv2.normalize(np.array(image), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255
        
        self.__featureImages.append(
            cv2.merge((norm(feature_1), norm(feature_2), norm(feature_3)))
            )

    def getFeatureImage(self):
        return self.__featureImages