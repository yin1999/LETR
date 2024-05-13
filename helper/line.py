#!/usr/bin/env python3
"""Process data for LETR
Usage:
    line.py <src> <dst>
    line.py (-h | --help )

Examples:
    python line.py line_raw line_processed

Arguments:
    <src>                Source directory that stores preprocessed line data
    <dst>                Temporary output directory

Options:
   -h --help             Show this screen.
"""

import json
import cv2
import os
import numpy as np
import enum
from docopt import docopt
import pickle
import matplotlib.pyplot as plt

class PathType(enum.IntEnum):
	# 无
	NONE = 0
	# 拐点
	KEYPOINT = 1

	#### 路径类型 ####
	# 直线
	LINE = 2
	# 圆弧
	ARC = 3
	# B 样条曲线
	BSPLINE = 4

	def __str__(self) -> str:
		return self.name

	def __repr__(self) -> str:
		return f"PathType.{self.name}"

	def __format__(self, format_spec: str) -> str:
		return self.__repr__()

	@classmethod
	def numberClasses(cls) -> int:
		return len(cls)

def main():
    args = docopt(__doc__)
    src_dir = args["<src>"]
    tar_dir = args["<dst>"]

    image_id = 0
    anno_id = 0
    for batch in ["train2017", "val2017"]:
        if batch == "train2017":
            anno_file = os.path.join(src_dir, "train.pkl")
        else:
            anno_file = os.path.join(src_dir, "valid.pkl")

        with open(anno_file, "rb") as f:
            dataset: list[list[tuple[float, float, PathType]]] = pickle.load(f)

        anno = {}
        anno['images'] = []
        anno['annotations'] = []

        idx = 1

        def handle(data: list[tuple[int, int, PathType]], image_id, anno_id):
            # remove the last channel
            data: list[tuple[int, int]] = [[int(x[0]), int(x[1])] for x in data]
            # create image from points
            # create a figure
            im = np.zeros((256, 256, 3), dtype=np.uint8)
            im.fill(255)
            # draw lines
            for i in range(len(data) - 1):
                cv2.line(im, (data[i][0], data[i][1]), (data[i + 1][0], data[i + 1][1]), (0, 0, 0), 1)

            filename = f"{image_id}.png"

            anno['images'].append({'file_name': filename, 'height': im.shape[0], 'width': im.shape[1], 'id': image_id})

            lines = []
            for i in range(len(data) - 1):
                lines.append([data[i], data[i + 1]])

            os.makedirs(os.path.join(tar_dir, batch), exist_ok=True)

            image_path = os.path.join(tar_dir, batch, filename)
            line_set = save_and_process(image_path, im, lines)
            for line in line_set:
                info = {}
                info['id'] = anno_id
                anno_id += 1
                info['image_id'] = image_id
                info['category_id'] = 0
                info['line'] = line
                info['area'] = 1
                anno['annotations'].append(info)

            print("Finishing", image_path)
            return anno_id
        
        anno['categories'] = [{'supercategory':"line", "id": "0", "name": "line"}]
        for lines in dataset:
            anno_id = handle(lines, image_id, anno_id)
            image_id += 1

        os.makedirs(os.path.join(tar_dir, "annotations"), exist_ok=True)
        anno_path = os.path.join(tar_dir, "annotations", f"lines_{batch}.json")
        with open(anno_path, 'w') as outfile:
            json.dump(anno, outfile)

            
def save_and_process(image_path, image, lines):
    # change the format from x,y,x,y to x,y,dx, dy
    # order: top point > bottom point
    #        if same y coordinate, right point > left point

    new_lines_pairs = []
    for line in lines: # [ #lines, 2, 2 ]
        p1 = line[0]    # xy
        p2 = line[1]    # xy
        if p1[0] < p2[0]:
            new_lines_pairs.append( [p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1]] ) 
        elif  p1[0] > p2[0]:
            new_lines_pairs.append( [p2[0], p2[1], p1[0]-p2[0], p1[1]-p2[1]] )
        else:
            if p1[1] < p2[1]:
                new_lines_pairs.append( [p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1]] )
            else:
                new_lines_pairs.append( [p2[0], p2[1], p1[0]-p2[0], p1[1]-p2[1]] )

    cv2.imwrite(image_path, image)
    return new_lines_pairs


if __name__ == "__main__":
    main()