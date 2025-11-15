import cv2
import numpy as np
from scipy.signal import butter
from scipy import signal
from tqdm import tqdm
from scipy import stats as st

MAX_OBJECTS = 1e9
MAX_FRAMES = 1e9

class VideoProcessor:
    __videoFileLocation:str = ""
    __getRoiCallback = None
    __raw_traces = None

    def __init__(self, videoFileLocation,  groundTruthLocation, getRoiCallback, fps = 30, isXmp = False) -> None:
        self.__videoFileLocation = videoFileLocation
        self.__groundTruthLocation = groundTruthLocation
        self.__getRoiCallback = getRoiCallback
        self.__fps = fps
        self.__isXmp = isXmp
        self.__groundTruthValue = []
        self.__groundTruthTrack = []
    
    def getFPS(self):
        return self.__fps
    
    def __trackInterpolate(self, gtTime, gtTrack):
        endTime = int(list(gtTime)[-1])
        x_interp = np.linspace(0, endTime, num=int(30*endTime/1000))
        gtTrackInptr = np.interp(x_interp, gtTime, gtTrack)
        gtTimeInptr = x_interp

        return gtTrackInptr
    
    def readVideo(self):
        cap = cv2.VideoCapture(self.__videoFileLocation)

        if self.__isXmp:
            gtdata = np.loadtxt(self.__groundTruthLocation, delimiter=',')
            gtTime = gtdata[:, 0]
            gtHR = self.__trackInterpolate(gtTime, gtdata[:, 1])
            gtTrack = self.__trackInterpolate(gtTime, gtdata[:, 3])
        else:
            gtdata = np.loadtxt(self.__groundTruthLocation)
            gtTime = gtdata[2, :]*1000
            gtHR = self.__trackInterpolate(gtTime, gtdata[1, :])
            gtTrack = self.__trackInterpolate(gtTime, gtdata[0, :])

        print(f"Reading video.. {self.__videoFileLocation} ")

        totFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pbar = tqdm(total=totFrame)

        self.__groundTruthValue = gtHR.copy()
        self.__groundTruthTrack = gtTrack.copy()
        self.__raw_traces = ()

        self.__raw_traces = {
            "r" : [],
            "g" : [],
            "b" : [],
            "y" : []
        }

        while(cap.isOpened()):
            is_read, frame = cap.read()
            
            if(is_read):
                frame = self.__getRoiCallback(frame)

                r, g, b, y = [], [], [], []

                meanCalc = self.meanImpl
                b.append(meanCalc(frame=frame[:, :, 0]))
                g.append(meanCalc(frame=frame[:, :, 1]))
                r.append(meanCalc(frame=frame[:, :, 2]))

                ycbcr=cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)                      

                y.append(meanCalc(frame=ycbcr[:, :, 0]))

                self.__raw_traces["r"].append(r)
                self.__raw_traces["g"].append(g)
                self.__raw_traces["b"].append(b)
                self.__raw_traces["y"].append(y)

                pbar.update(1)
            else:
                break

        pbar.close()
        cap.release()

    def getAllFrames(self):
        return self.__roiedFrames    
    def getHR(self):
        return self.__groundTruthValue
    def getTrack(self):
        return self.__groundTruthTrack
    
    def meanImpl(self, frame):
        nz = frame.ravel()
        nz = nz[nz != 0]
        return float(nz.mean()) if nz.size else 0.0
    
    def getRawTraces(self):
        return self.__raw_traces
    
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

class ChormFeatures:
    __chunks:list = []

    def __init__(self, videoFeature: VideoProcessor, stride, order=2, window_length=128, meanFoo=None):
        self.__videoFeature = videoFeature
        self.__order = order
        self.__fps = videoFeature.getFPS()
        self.__count = 0
        self.__featureImages = []
        self.length = window_length
        self.__stride = stride
        self.__meanFoo = meanFoo

        self.__chunks = self.__videoFeature.getChuncks(self.__stride, window_length, self.__meanFoo)

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

    def __buildCHROM(self, chunk):
        length = self.length

        r = chunk["r"]
        g = chunk["g"]
        b = chunk["b"]
        y = chunk["y"]
        hr = chunk["hr"]
        ppg = chunk["ppg"]

        tempGtHr = hr
        tempGtTrack = ppg

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
        for chunk in self.__chunks:
            self.__buildCHROM(chunk)

    def getFeatureImage(self):
        return self.__featureImages
