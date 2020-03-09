from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import itertools
import numpy as np
import cv2
import re
from tqdm import tqdm
from data_utils import total_files_counter, walk_dir



THRESH_VALUE = 120

MAX_THRESH_VALUE = 255

MIN_CNTR_HUMN_AREA = 8
MAX_CNTR_HUMN_AREA = 350

def human_detection_simple(inp_grayscalesPth, inp_heatmapsPth, det_outImgsPth, upsample_ratio = None):
    
    
    heatmapsDetDir = os.path.join(det_outImgsPth, )
    graysDetDir = os.path.join(det_outImgsPth, 'det_grays')
    
    if not os.path.exists(inp_grayscalesPth):
        print (inp_grayscalesPth + ' is not a valid path')
        exit(-1)
    
    if not os.path.exists(inp_heatmapsPth):
        print (inp_heatmapsPth + ' is not a valid path')
   
    image_counter = total_files_counter(inp_grayscalesPth, '.png')
   
    if not image_counter:
        print (inp_grayscalesPth + ' contains no images')
        exit(-1)
    
    if not os.path.exists(graysDetDir):
        os.makedirs(graysDetDir)
    
    if not os.path.exists(heatmapsDetDir):
        os.makedirs(heatmapsDetDir)
    
    for grayscale, heatmap in tqdm(itertools.izip(walk_dir(inp_grayscalesPth,'.png'),
                                             walk_dir(inp_heatmapsPth, '.png')),
                                             total=image_counter, desc = 'Generating human detection images'):
       
        grayscale_img = cv2.imread(grayscale,cv2.IMREAD_GRAYSCALE)
        heatmap_img = cv2.imread(heatmap)
        
        ret, thresh = cv2.threshold(grayscale_img, THRESH_VALUE, MAX_THRESH_VALUE, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        areas = [cv2.contourArea(c) for c in contours]
        for idx, val in enumerate(areas):
            
            if MIN_CNTR_HUMN_AREA <= val <= MAX_CNTR_HUMN_AREA:
                cntr = contours[idx]
                
                x,y,w,h = cv2.boundingRect(cntr)
                
                xmin = x
                ymin = y
                xmax = x+w
                ymax = y+h
               
                cv2.rectangle(grayscale_img,(xmin,ymin),(xmax,ymax),(0,0,255),1)
                cv2.rectangle(heatmap_img,(xmin,ymin),(xmax,ymax),(0,0,255),1)
       
        if upsample_ratio is not None:
            grayscale_img = cv2.resize(grayscale_img, (upsample_ratio*grayscale_img.shape[1],
                                       upsample_ratio*grayscale_img.shape[0]),
                                       interpolation = cv2.INTER_NEAREST)
            heatmap_img = cv2.resize(heatmap_img, (upsample_ratio*heatmap_img.shape[1],
                                     upsample_ratio*heatmap_img.shape[0]),
                                     interpolation = cv2.INTER_NEAREST)
        
        cv2.imwrite(os.path.join(graysDetDir, (os.path.splitext(os.path.basename(grayscale))[0] + '.png')), grayscale_img)
        cv2.imwrite(os.path.join(heatmapsDetDir, (os.path.splitext(os.path.basename(heatmap))[0] + '.png')), heatmap_img)
    return graysDetDir, heatmapsDetDir