from FeatureProcessingBHrPPG.main import *
from FeatureProcessing import roi
import matplotlib.pyplot as plt
import numpy as np

# video = VideoFeature("D:\\UBFC Dataset\\subject1\\vid.avi", "D:\\UBFC Dataset\\subject1\\ground_truth.txt", getRoiCallback=roi.getROI, maxObjects=50)

videoDir = "D:\\D-Download\\Pub_BH-rPPG_FULL\\Pub_BH-rPPG_FULL\\0_0"
videoName = "0_0"

video = VideoFeature(videoDir, videoName, roi.getROI, maxObjects=10)

video.readVideo()

chrom = ChormFeatures(video)
chrom.buildCHROM()

def DFT(signal_, plot = True):
    N = len(signal_)

    yf = np.fft.rfft(signal_)
    xf = np.fft.rfftfreq(N, 1/30)

    if(plot):
        plt.plot(xf, np.abs(yf))
        plt.xlabel('Frequency (Hz)')
        plt.show()

for i in range(10):
    # print(chrom.getFeatureImage()[i][1].mode/60)

    plt.plot(chrom.getFeatureImage()[i][2])
    plt.show()
    DFT(chrom.getFeatureImage()[0][2])
