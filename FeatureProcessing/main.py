import cv2
import numpy as np

class VideoFeature:
    __videoFileLocation:str = ""
    __maxFrameLength:int = 0
    __getRoiCallback:function = None
    __rgbMatrix:np.ndarray = None

    def __init__(self, videoFileLocation, getRoiCallback, maxFrameLength = -1) -> None:
        self.__videoFileLocation = videoFileLocation
        self.__maxFrameLength = maxFrameLength
        self.__getRoiCallback = getRoiCallback

        self.__ReadVideo()
    
    def __ReadVideo(self):
        cap = cv2.VideoCapture(self.__videoFileLocation)
        frameCount = 0
        b = [], g = [], r = []
        
        while(cap.isOpened() and (self.__maxFrameLength == -1 or frameCount < self.maxFrameLength)):
            is_read, frame = cap.read()
            
            if(is_read):
                frame = self.__getRoiCallback(frame)

                b.append(np.mean(frame[:, :, 0]))
                g.append(np.mean(frame[:, :, 1]))
                r.append(np.mean(frame[:, :, 2]))

                frameCount += 1

        cap.release()
        self.__rgbMatrix = np.array([r, g, b])

    