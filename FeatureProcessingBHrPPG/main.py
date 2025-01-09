import cv2
import numpy as np
from scipy.signal import butter
from scipy import signal
from tqdm import tqdm
from scipy import stats as st
import os
import pandas as pd

MAX_OBJECTS = 1e9
MAX_FRAMES = 1e9

class VideoFeature:
    __maxFrameLength:int = 0
    __getRoiCallback = None
    __colorMatrix:list[np.ndarray] = None

    def __init__(self, videoDir, videoName, getRoiCallback, maxFrameLength = 128, fps = 30, maxObjects = 10) -> None:
        self.__videoDir = videoDir
        self.__getRoiCallback = getRoiCallback
        self.__maxObjects = maxObjects
        self.__maxFrameLength = maxFrameLength
        self.__videoName = videoName
        self.__fps = fps
        self.__colorMatrix = []
        self.__groundTruthValue = []
        self.__groundTruthTrack = []
    
    def getFPS(self):
        return self.__fps

    def setMaxVideoLength(self, maxFrameLength):
        self.__maxFrameLength = maxFrameLength
    
    def __trackInterpolate(self, timeStamp, wave):
        endTime = int(timeStamp[-1])
        x_interp = np.linspace(0, endTime, num=900)
        waveTimeStamp = np.linspace(0, endTime, num=len(wave))
        gtTrackInptr = np.interp(x_interp, waveTimeStamp, wave)

        print(len(gtTrackInptr))

        return (x_interp, gtTrackInptr)
    
    def readVideo(self):
        # gtdata = np.loadtxt(self.__groundTruthLocation, delimiter=',')
        # gtTime = gtdata[:, 0]
        # gtHR = self.__trackInterpolate(gtTime, gtdata[:, 1])
        # gtTrack = self.__trackInterpolate(gtTime, gtdata[:, 3])

        ts = pd.read_csv(f"{self.__videoDir}/timestamps.csv", header=None)[0].tolist()
        wv = pd.read_csv(f"{self.__videoDir}/wave.csv")["Wave"].tolist()

        timeStamp, wave = self.__trackInterpolate(ts, wv)
        
        self.__colorMatrix.clear()

        frameCount = 0
        objectCount = 0
        ms = 0
        frameWindow = [0, 0]

        b = []
        g = []
        r = []
        y = []
        
        photosDir = self.__videoDir + "\\" +self.__videoName
        photosNameList = sorted(os.listdir(photosDir))

        print(f"Reading video.. {photosDir}")

        totFrame = 900
        tdqmTotal = min(totFrame-self.__maxFrameLength, self.__maxObjects)
        
        pbar = tqdm(total=tdqmTotal)

        for photoLoc in photosNameList:
            frame = cv2.imread(f"{photosDir}/{photoLoc}", cv2.IMREAD_COLOR)

            frame = self.__getRoiCallback(frame)

            b.append(np.mean(frame[:, :, 0]))
            g.append(np.mean(frame[:, :, 1]))
            r.append(np.mean(frame[:, :, 2]))

            ycbcr=cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)                      

            y.append(np.mean(ycbcr[:, :, 0]))

            frameCount += 1
            ms += 1
            frameWindow[1] += 1

            if(frameCount >= self.__maxFrameLength):
                # e_time = (ms//30)*1000
                # s_time = e_time - (self.__maxFrameLength//30)*1000

                # s_idx = np.searchsorted(gtTime, s_time)
                # e_idx = np.searchsorted(gtTime, e_time)

                s_idx, e_idx = frameWindow

                if(e_idx >= len(wave)):
                    break

                self.__colorMatrix.append(np.array([r, g, b, y]))

                self.__groundTruthValue.append(0)

                self.__groundTruthTrack.append(
                    wave[s_idx:e_idx]
                )

                frameCount = frameCount-1
                objectCount += 1

                r = r[1:]
                g = g[1:]
                b = b[1:]
                y = y[1:]

                frameWindow[0] += 1

                pbar.update(1)

    def isValidIndex(self, index):
        return index < len(self.__colorMatrix)
    
    def getColors(self, index):
        if index >= len(self.__colorMatrix):
            return None
        return self.__colorMatrix[index]
    
    def getGroundTruth(self, index):
        if index >= len(self.__colorMatrix):
            return None
        return (self.__groundTruthValue[index], self.__groundTruthTrack[index])
    
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

    def __buildCHROM(self):     
        length = self.__videoFeature.getMaxFrameLength()

        r = self.__videoFeature.getColors(self.__count)[0]
        g = self.__videoFeature.getColors(self.__count)[1]
        b = self.__videoFeature.getColors(self.__count)[2]
        y = self.__videoFeature.getColors(self.__count)[3]
        tempGtHr, tempGtTrack = self.__videoFeature.getGroundTruth(self.__count)

        self.__count += 1
        
        r /= np.mean(r)*100
        g /= np.mean(g)*100
        b /= np.mean(b)*100
        y_norm = y/np.mean(y)*100

        r_ = self.__butterBandpassFilter(r, 0.7, 5)
        g_ = self.__butterBandpassFilter(g, 0.7, 5)
        b_ = self.__butterBandpassFilter(b, 0.7, 5)
        y_norm = self.__butterBandpassFilter(y, 0.7, 5)

        self.__X = 3*r_ - 2*g_
        self.__Y = 1.5*r_ + g_ - 1.5*b_
        self.__Y_lum = y

        feature_1 = []
        feature_2 = []
        feature_3 = []

        for i in range(length//2):
            feature_1.append(self.__X[i : i+(length//2)])
            feature_2.append(self.__Y[i : i+(length//2)])
            feature_3.append(self.__Y_lum[i : i+(length//2)])
        
        def norm(image):
            return cv2.normalize(np.array(image), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255
        
        tempImage = cv2.merge((norm(feature_1), norm(feature_2), norm(feature_3)))
        rawColors = np.array([r_, g_, b_, y, y_norm])

        self.__featureImages.append((tempImage, tempGtHr, tempGtTrack, rawColors))
            
    def buildCHROM(self):
        while(self.__videoFeature.isValidIndex(self.__count)):
            self.__buildCHROM()

    def getFeatureImage(self):
        return self.__featureImages
