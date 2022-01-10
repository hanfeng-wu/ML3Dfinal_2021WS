from numpy.lib.arraypad import pad
import torch
import torch.nn as nn
import numpy as np


class Vertex:
    def __init__(self, num):
        pass



def isPositionValid(pos):
	return not (pos[0] == -float('inf') or  pos[1] == -float('inf') or  pos[2] == -float('inf'))

def isThresholdSatisfied(pos1, pos2, edgeThreshold):
	return np.sqrt(np.sum((pos1-pos2)**2)) <= edgeThreshold

def isValid(positions, n1, n2, n3, edgeThreshold):
    pos1 = positions[n1]
    pos2 = positions[n2]
    pos3 = positions[n3]
    if (not isPositionValid(pos1) or not isPositionValid(pos2) or not isPositionValid(pos3)):
        return False
    else:
        if (not isThresholdSatisfied(pos1,pos2,edgeThreshold) or not isThresholdSatisfied(pos2,pos3,edgeThreshold) or not isThresholdSatisfied(pos3,pos1,edgeThreshold)):
            return False
        else:
            return True
		




class Surface:
    def __init__(self, number, indice):
        self.number = number
        self.indice = indice

            
        

class RGBD2Mesh:
    def __init__(self, depthMap, height, width, depthIntrinsics, depthExtrinsics):
        self.depthMap = depthMap
        self.depthIntrinsics = depthIntrinsics
        self.height = height
        self.width = width
        #self.trajectory = trajectory
        self.depthExtrinsics = depthExtrinsics

    def writemesh(self, positions, filepath):
        edgeThreshold = 0.01
        nVertices = self.width * self.height
        nFaces = 0
        f = open(filepath, 'w')
        if(not f): return False

        f.write('COFF\n')
        f.write(f'{positions.shape[0]} {nFaces} 0\n')
        for pos in positions:
            f.write(f'{pos[0]} {pos[1]} {pos[2]} \n')

        f.close()
        return True            


    def tomesh(self):
        fX = self.depthIntrinsics[0][0]
        fY = self.depthIntrinsics[1][1]
        cX = self.depthIntrinsics[0][2]
        cY = self.depthIntrinsics[1][2]
        depthExtrinsicsInv = np.linalg.inv(self.depthExtrinsics)
        indexinf = []
        positions = []
        num = self.height * self.width
        #trajectoryInv = np.linalg.inv(self.trajectory)
        for i in range(num):
            depth = self.depthMap[i]
            ux = int(i % self.width)
            uy = int(i / self.width)
            if (depth > 1000):
                pass    
            else:
                x = (ux - cX) / fX
                y = (uy - cY) / fY
                worldSpacePosition = depthExtrinsicsInv @ np.array([depth*x, depth*y, depth, 1.0])
                worldSpacePosition /= worldSpacePosition[3]
                positions.append(worldSpacePosition)
        positions = np.array(positions)
        self.writemesh(positions, './testmesh.off')        





        