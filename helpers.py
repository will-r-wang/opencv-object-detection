# WIP implementation of indidivdual object tracking and speed determination

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False

class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
