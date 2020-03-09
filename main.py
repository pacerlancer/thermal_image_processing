from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
from sys import argv
from argparse import ArgumentParser
from data_utils import generate_dataset_images, generate_detctions_video
from human_detector import human_detection_simple


def main(args):
    
    sub_dirs_path = glob.glob(args.input_dir+'*')
    for folderpath in sub_dirs_path:
       
        sub_fldr_name = os.path.splitext(os.path.basename(folderpath))[0]
        print(sub_fldr_name + '...')
        print('===========================')
       
        sub_results_dir = os.path.join(args.output_dir, sub_fldr_name)
        if not os.path.exists(sub_results_dir):
            os.makedirs(sub_results_dir)
       
        thermalImgsDir, grayImgsDir, heatmapImgsDir = generate_dataset_images(folderpath, sub_results_dir)
       
        graysDetDir, heatmapsDetDir = human_detection_simple(grayImgsDir, heatmapImgsDir, sub_results_dir, args.upsample_ratio)
        
        generate_detctions_video(graysDetDir, heatmapsDetDir, sub_results_dir, sub_fldr_name)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', '-i',
                        help='input data folder which contains sub-folder(s) of .txt raw radiometry data images',
                        default=os.environ.get('INPUT_DIR', '/dataset/FLIR_data/'))
    parser.add_argument('--upsample_ratio', '-u', type=int, default=None,
                        help='the factor (shall be equal or larger than 1)to upsample the resolution of the generated video images\
                        results with for better visualisation')
    parser.add_argument('--output_dir', '-o',
                        help='output folder for the generated dataset images and the detection results',
                        default=os.environ.get('OUTPUT_DIR', '/dataset/FLIR_data/results/'))
    args = parser.parse_args()
    main(args)