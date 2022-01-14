from numpy.lib.arraypad import pad
import torch
import torch.nn as nn
import numpy as np
from Networks.obj2pointcloud import *

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
                worldSpacePosition = np.array((worldSpacePosition[0],worldSpacePosition[2],-worldSpacePosition[1]))
                positions.append(worldSpacePosition)
        positions = np.array(positions)
        #self.writemesh(positions, './testmesh.off')
        return positions  


    def create_faces(self, filepath):
        fX = self.depthIntrinsics[0][0]
        fY = self.depthIntrinsics[1][1]
        cX = self.depthIntrinsics[0][2]
        cY = self.depthIntrinsics[1][2]
        depthExtrinsicsInv = np.linalg.inv(self.depthExtrinsics)

        positions = np.zeros((self.height * self.width, 4))
        num = self.height * self.width
        #trajectoryInv = np.linalg.inv(self.trajectory)
        for i in range(num):
            depth = self.depthMap[i]
            ux = int(i % self.width)
            uy = int(i / self.width)
            if (depth > 1000):
                positions[i] = np.array([-float('inf'),-float('inf'),-float('inf'),-float('inf')])
            else:
                x = (ux - cX) / fX
                y = (uy - cY) / fY
                worldSpacePosition = depthExtrinsicsInv @ np.array([depth*x, depth*y, depth, 1.0])
                worldSpacePosition /= worldSpacePosition[3]
                worldSpacePosition = np.array((worldSpacePosition[0],worldSpacePosition[2],-worldSpacePosition[1], worldSpacePosition[3]))
                positions[i] = worldSpacePosition


        edgeThreshold = 0.05
        surfaces = []
        nVertices = self.width * self.height
        for i in range(self.height-1):
            for j in range(self.width-1):
                i1 = i
                j1 = j
                i2 = i+1
                j2 = j
                i3 = i
                j3 = j+1
                i4 = i+1
                j4 = j+1
                try:
                    if (isValid(positions, i1*self.width+j1, i2*self.width+j2, i3*self.width+j3, edgeThreshold)):
                        surfaces.append(Surface(3, np.array([i1*self.width+j1, i2*self.width+j2, i3*self.width+j3])))
                except(IndexError):
                    print(f'i:{i}, j:{j}, width:{self.width}, height:{self.height}, {len(positions)}')
                    
                if (isValid(positions, i2*self.width+j2, i4*self.width+j4, i3*self.width+j3, edgeThreshold)):
                    surfaces.append(Surface(3, np.array([i2*self.width+j2, i4*self.width+j4, i3*self.width+j3])))
        nFaces = len(surfaces)
        f = open(filepath, 'w')
        if(not f): return False

        f.write('COFF\n')
        f.write(f'{nVertices} {nFaces} 0\n')
        for i in range(self.height):
            for j in range(self.width):
                pos = positions[i*self.width+j]
                if (pos[0] == -float('inf') or pos[1] == -float('inf') or pos[2] == -float('inf')):
                    f.write('0 0 0 \n')
                else:
                    f.write(f'{pos[0]} {pos[1]} {pos[2]} \n')
        for i in range(nFaces):
            f.write(f'{surfaces[i].number} ')
            for j in range(len(surfaces[i].indice)):
                if(j != len(surfaces[i].indice)-1):
                    f.write(f'{surfaces[i].indice[j]} ')
                else:
                    f.write(f'{surfaces[i].indice[j]}\n')
        f.close()
        return True

    def to_pcd(self, numpoints):
        fX = self.depthIntrinsics[0][0]
        fY = self.depthIntrinsics[1][1]
        cX = self.depthIntrinsics[0][2]
        cY = self.depthIntrinsics[1][2]
        depthExtrinsicsInv = np.linalg.inv(self.depthExtrinsics)

        positions = np.zeros((self.height * self.width, 4))
        num = self.height * self.width
        #trajectoryInv = np.linalg.inv(self.trajectory)
        for i in range(num):
            depth = self.depthMap[i]
            ux = int(i % self.width)
            uy = int(i / self.width)
            if (depth > 1000):
                positions[i] = np.array([-float('inf'),-float('inf'),-float('inf'),-float('inf')])
            else:
                x = (ux - cX) / fX
                y = (uy - cY) / fY
                worldSpacePosition = depthExtrinsicsInv @ np.array([depth*x, depth*y, depth, 1.0])
                worldSpacePosition /= worldSpacePosition[3]
                worldSpacePosition = np.array((worldSpacePosition[0],worldSpacePosition[2],-worldSpacePosition[1], worldSpacePosition[3]))
                positions[i] = worldSpacePosition


        edgeThreshold = 0.05
        surfaces = []
        for i in range(self.height-1):
            for j in range(self.width-1):
                i1 = i
                j1 = j
                i2 = i+1
                j2 = j
                i3 = i
                j3 = j+1
                i4 = i+1
                j4 = j+1
                try:
                    if (isValid(positions, i1*self.width+j1, i2*self.width+j2, i3*self.width+j3, edgeThreshold)):
                        surfaces.append(Surface(3, np.array([i1*self.width+j1, i2*self.width+j2, i3*self.width+j3])))
                except(IndexError):
                    print(f'i:{i}, j:{j}, width:{self.width}, height:{self.height}, {len(positions)}')
                    
                if (isValid(positions, i2*self.width+j2, i4*self.width+j4, i3*self.width+j3, edgeThreshold)):
                    surfaces.append(Surface(3, np.array([i2*self.width+j2, i4*self.width+j4, i3*self.width+j3])))

        vertices = []


        for i in range(self.height):
            for j in range(self.width):
                pos = positions[i*self.width+j]
                if (pos[0] == -float('inf') or pos[1] == -float('inf') or pos[2] == -float('inf')):
                    vertices.append([0,0,0])
                else:
                    vertices.append([pos[0], pos[1], pos[2]])


        faces = [s.indice for s in surfaces]

        vertices = np.array(vertices)
        faces = np.array(faces)

        pcd = sample_point_cloud(vertices, faces, numpoints)


        return pcd

  





        