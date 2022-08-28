from xml.etree.ElementTree import ElementTree
import numpy as np 
from warnings import warn 
import itertools 

STATE_UNSET = 0 
STATE_INTERP = 1 
STATE_LABELED = 2 

stateFromText = {
    '0': STATE_UNSET,
    '1': STATE_INTERP,
    '2': STATE_LABELED,
}

OCC_UNSET = 255 
OCC_VISIBLE = 0
OCC_PARTLY = 1 
OCC_FULLY = 2 

occFromText = {
    '-1': OCC_UNSET,
    '0': OCC_VISIBLE,
    '1': OCC_PARTLY,
    '2': OCC_FULLY,
}

TRUNC_UNSET = 255 
TRUNC_IN_IMAGE = 0
TRUNC_TRUNCATED = 1 
TRUNC_OUT_IMAGE = 2 
TRUNC_BEHIND_IMAGE = 3 

truncFromText = {
    '99': TRUNC_UNSET,
    '0': TRUNC_IN_IMAGE,
    '1': TRUNC_TRUNCATED,
    '2': TRUNC_OUT_IMAGE,
    '3': TRUNC_BEHIND_IMAGE,
}


class Tracklet(object):

    objectType = None 
    size = None # float array (height, width, length) 
    firstFrame = None 
    trans = None # nx3 float array (x, y, z) 
    rots = None # nx3 float array (x, y, z) 
    states = None # len-n uint8 array of states  
    occs = None # nx2 uint8 array (occlusion, occlusion_kf) 
    truncs = None # len-n uint8 array of truncation 
    amtOccs = None # None or (nx2) float array (amt_occlusion, amt_occlusion_kf) 
    amtBorders = None # None (nx3) float array (amt_border_l, amt_border_r, amt_border_kf) 
    nFrames = None


    def __init__(self):
        self.size = np.nan * np.ones(3, dtype=float)


    def __str__(self):
        return "[Tracklet over {0} frames for {1}]".format(self.nFrames, self.objectType)

    
    def __iter__(self):
        if self.amtOccs is None: 
            return zip(self.trans, self.rots, self.states, self.occs, self.truncs, 
                    itertools.repeat(None), itertools.repeat(None), 
                    range(self.firstFrame, self.firstFrame + self.nFrames))
        else:
            return zip(self.trans, self.rots, self.states, self.occs, self.truncs, 
                    self.amtOccs, self.amtBorders, range(self.firstFrame, self.firstFrame + self.nFrames))
    


def parseXML(trackletFile):
    eTree = ElementTree()
    print("Parsing tracklet file: {}".format(trackletFile))
    
    with open(trackletFile) as f: 
        eTree.parse(f)

    # convert the output to a list of Tracklets 
    trackletsElem = eTree.find('tracklets')
    tracklets = []
    trackletIdx = 0 
    nTracklets = None 

    for trackletElem in trackletsElem:
        if trackletElem.tag == 'count':
            nTracklets = int(trackletElem.text)
            print("File contains {} tracklets".format(nTracklets))
        elif trackletElem.tag == 'item_version':
            pass 
        elif trackletElem.tag == 'item':
            newTrack = Tracklet()
            isFinished = False 
            hasAmt = False 
            frameIdx = None 

            for info in trackletElem:
                if isFinished:
                    raise ValueError('more info on element after finished')
                if info.tag == 'objectType':
                    newTrack.objectType = info.text 
                elif info.tag == 'h':
                    newTrack.size[0] = float(info.text)
                elif info.tag == 'w':
                    newTrack.size[1] = float(info.text)
                elif info.tag == 'l':
                    newTrack.size[2] = float(info.text)
                elif info.tag == 'first_frame':
                    newTrack.firstFrame = int(info.text)
                elif info.tag == 'poses':
                    for pose in info:
                        if pose.tag == 'count':
                            if newTrack.nFrames is not None:
                                raise ValueError('There are several pose lists for a single track')
                            elif frameIdx is not None:
                                raise ValueError('?')
                            newTrack.nFrames = int(pose.text)
                            newTrack.trans = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
                            newTrack.rots = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
                            newTrack.states = np.nan * np.ones(newTrack.nFrames, dtype='uint8')
                            newTrack.occs = np.nan * np.ones((newTrack.nFrames, 2), dtype='uint8')
                            newTrack.truncs = np.nan * np.ones(newTrack.nFrames, dtype='uint8')
                            newTrack.amtOccs = np.nan * np.ones((newTrack.nFrames, 2), dtype=float)
                            newTrack.amtBorders = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
                            frameIdx = 0
                        elif pose.tag == 'item_version':
                            pass
                        elif pose.tag == 'item':
                            # pose in one frame
                            if frameIdx is None:
                                raise ValueError('pose item came before number of poses!')
                            for poseInfo in pose:
                                if poseInfo.tag == 'tx':
                                    newTrack.trans[frameIdx, 0] = float(poseInfo.text)
                                elif poseInfo.tag == 'ty':
                                    newTrack.trans[frameIdx, 1] = float(poseInfo.text)
                                elif poseInfo.tag == 'tz':
                                    newTrack.trans[frameIdx, 2] = float(poseInfo.text)
                                elif poseInfo.tag == 'rx':
                                    newTrack.rots[frameIdx, 0] = float(poseInfo.text)
                                elif poseInfo.tag == 'ry':
                                    newTrack.rots[frameIdx, 1] = float(poseInfo.text)
                                elif poseInfo.tag == 'rz':
                                    newTrack.rots[frameIdx, 2] = float(poseInfo.text)
                                elif poseInfo.tag == 'state':
                                    newTrack.states[frameIdx] = stateFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion':
                                    newTrack.occs[frameIdx, 0] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion_kf':
                                    newTrack.occs[frameIdx, 1] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'truncation':
                                    newTrack.truncs[frameIdx] = truncFromText[poseInfo.text]
                                elif poseInfo.tag == 'amt_occlusion':
                                    newTrack.amtOccs[frameIdx,0] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_occlusion_kf':
                                    newTrack.amtOccs[frameIdx,1] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_border_l':
                                    newTrack.amtBorders[frameIdx,0] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_border_r':
                                    newTrack.amtBorders[frameIdx,1] = float(poseInfo.text)
                                    hasAmt = True
                                elif poseInfo.tag == 'amt_border_kf':
                                    newTrack.amtBorders[frameIdx,2] = float(poseInfo.text)
                                    hasAmt = True
                                else:
                                    raise ValueError('unexpected tag in poses item: {}!'.format(poseInfo.tag))
                            frameIdx += 1
                        else:
                            raise ValueError('unexpected pose info: {}!'.format(pose.tag))
                elif info.tag == 'finished':
                    isFinished = True
                else:
                    raise ValueError('unexpected tag in tracklets: {}!'.format(info.tag))
            #end: for all fields in current tracklet

            # some final consistency checks on new tracklet
            if not isFinished:
                warn('tracklet {0} was not finished!'.format(trackletIdx))
            if newTrack.nFrames is None:
                warn('tracklet {0} contains no information!'.format(trackletIdx))
            elif frameIdx != newTrack.nFrames:
                warn('tracklet {0} is supposed to have {1} frames, but perser found {1}!'.format(trackletIdx, 
                    newTrack.nFrames, frameIdx))
            if np.abs(newTrack.rots[:,:2]).sum() > 1e-16:
                warn('track contains rotation other than yaw!')

            # if amtOccs / amtBorders are not set, set them to None
            if not hasAmt:
                newTrack.amtOccs = None
                newTrack.amtBorders = None

            # add new tracklet to list
            tracklets.append(newTrack)
            trackletIdx += 1

        else:
            raise ValueError('unexpected tracklet info')

    print('Loaded {} tracklets.'.format(trackletIdx))

    # final consistency check
    if trackletIdx != nTracklets:
        warn('according to xml information the file has {0} tracklets, but parser found {1}!'.format(nTracklets, trackletIdx))

    return tracklets


