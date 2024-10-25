from FeatureProcessing.main import *
from FeatureProcessing import roi

video = VideoFeature("D:\\UBFC Dataset\\subject1\\vid.avi", "D:\\UBFC Dataset\\subject1\\ground_truth.txt", getRoiCallback=roi.getROI, maxObjects=2)

video.readVideo()
