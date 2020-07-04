# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:16:08 2020

@author: ryanc
"""

#the purpose of this script is to implement the ssd computer vision algorithm
#I am going to do this professionally in MATLAB but here i think it will be
#useful to increase my intuition and choose a better implementation for my
#particular use case. 


import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
import imageio

#so i'm not going to get all these libraries working. I tried for 10 mins already
#and i'm not even going to use this language so let's rock

#defining the model that will do the detections

def detect(frame,net,transform):
    height,width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2,0,1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width,height])
    # detections = [batch,number of classes,number of occurences,(score,x0,y0,x1,y1)]
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6:
            pt = (detections[0,i,j,1:] * scale).numpy()
            cv2.rectangle(frame,(int(pt[0]),int(pt[1])),(int(pt[2]),int(pt[3])),(255,0,0),2)
            cv2.putText(frame,labelmap[i - 1],(int(pt[0]),int(pt[1])),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
            j += 1
    return frame


#creating the SSD nerual network.
net = build_ssd('test')
net.load_state_dict(torch.load('path_to_file/', map_location = lambda storage, loc:storage))

#creating the transformation
transform = BaseTransform(net.size,(104/256.0, 117/256.9, 123/256.0))

#doing some object detection on a video 
reader = imageio.get_reader('path_to_video')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4',fps = fps)
for i, frame in enumerate(reader):
    detect(frame,net.eval(),transform)
            
    