import cv2
import numpy as np
from scipy.signal import butter
from scipy import signal
from tqdm import tqdm
from scipy import stats as st
import os
import pandas as pd

class BHVideoFeature:
    __getRoiCallback = None

    def __init__(self, videoDir, videoName, getRoiCallback, maxFrameLength = 128, fps = 30, maxObjects = 10) -> None:
        self.__videoDir = videoDir
        self.__getRoiCallback = getRoiCallback
        self.__videoName = videoName
        self.__fps = fps
        
    def getFPS(self):
        return self.__fps
    
    def __trackInterpolate(self, timeStamp, wave):
        endTime = int(timeStamp[-1])
        x_interp = np.linspace(0, endTime, num=900*2)
        waveTimeStamp = np.linspace(0, endTime, num=len(wave))
        gtTrackInptr = np.interp(x_interp, waveTimeStamp, wave)

        print(len(gtTrackInptr))

        return (x_interp, gtTrackInptr)
    
    def meanImpl(self, frame):
        nz = frame.ravel()
        nz = nz[nz != 0]
        return float(nz.mean()) if nz.size else 0.0
    
    def readVideo(self):
        ts = pd.read_csv(f"{self.__videoDir}/timestamps.csv", header=None)[0].tolist()
        wv = pd.read_csv(f"{self.__videoDir}/wave.csv")["Wave"].tolist()

        timeStamp, wave = self.__trackInterpolate(ts, wv)
        self.__groundTruthTrack = np.array(wave)

        self.__raw_traces = {
            "r" : [],
            "g" : [],
            "b" : [],
            "y" : []
        }
        
        photosDir = self.__videoDir + "\\" +self.__videoName
        photosNameList = sorted(os.listdir(photosDir))

        print(f"Reading video.. {photosDir}")

        totFrame = 900
    
        pbar = tqdm(total=totFrame)

        for photoLoc in photosNameList:
            frame = cv2.imread(f"{photosDir}/{photoLoc}", cv2.IMREAD_COLOR)

            frame = self.__getRoiCallback(frame)

            self.__raw_traces["b"].append(self.meanImpl(frame=frame[:, :, 0]))
            self.__raw_traces["g"].append(self.meanImpl(frame=frame[:, :, 1]))
            self.__raw_traces["r"].append(self.meanImpl(frame=frame[:, :, 2]))

            ycbcr=cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)                      

            self.__raw_traces["y"].append(self.meanImpl(frame=ycbcr[:, :, 0]))

            pbar.update(1)
    
    def getTrack(self):
        return self.__groundTruthTrack

    def getChuncks(self, stride:int, chunk_size = 128):
        r, g, b, y = self.__raw_traces["r"], self.__raw_traces["g"], self.__raw_traces["b"], self.__raw_traces["y"]

        chunks = []
        
        for start in range(0, len(b) - chunk_size + 1, stride):
            end = start + chunk_size

            chunk = {
                "r": r[start:end],
                "g": g[start:end],
                "b": b[start:end],
                "y": y[start:end],
                "hr": self.__groundTruthValue[start:end],
                "ppg": self.__groundTruthTrack[start:end]
            }
            chunks.append(chunk)
        
        return chunks
