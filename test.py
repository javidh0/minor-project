from FeatureProcessing.main import *
from FeatureProcessing import roi

video = VideoFeature("C:\\Users\\JAVIDH S\\Downloads\\vid.avi", "C:\\Users\\JAVIDH S\\Downloads\\gtdump (1).xmp", getRoiCallback=roi.getROI, maxObjects=2, isXmp=True)

video.readVideo()
