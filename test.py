from FeatureProcessing.main import *
from FeatureProcessing import roi

video = VideoFeature("C:\\Users\\JAVIDH S\\Downloads\\vid.avi", getRoiCallback=roi.getROI, maxObjects=2)

video.readVideo()
