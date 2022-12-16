import cv2
import glob
import os
import random
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import progressbar
import math
from generate_image import *
from utils import *
from aug import augmention

available_number = [x.replace("\n", "") for x in open('classes_num.txt').readlines()]
# available_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
available_char = [x.replace("\n", "") for x in open('classes_char.txt').readlines()]
# available_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
available_all = available_number + available_char
available_template = ['NN-CN/NNNN', 'NN-CN/NNN.NN', 'NNC/NNN.NN', 'NNC/NNNN', 'NNC-NNNN', 'NNC-NNN.NN']
# available_template = ['**-**/****', '**-**/***.**', '***/***.**', '***/****', '***-****', '***-***.**']
available_square_bg = glob.glob('background/square*.jpg')
available_rec_bg = glob.glob('background/rec*.jpg')

total_template = len(available_template)
total_number = len(available_number)
total_char = len(available_char)

visual = False

data = open('classes.txt', 'r').read().strip().split('\n')
box_label = dict()
for i in range(len(data)):
	box_label[data[i]] = i

assert os.path.exists('classes.txt') == True, 'Not exists file classes.txt, try again !'

def generate_boundingbox(sample, template, background, textsize, size = (480, 400), margin = 10):
	if '/' in template:
		return generate_2lines_boundingbox(sample, template, background, textsize)
	else:
		return generate_1line_boundingbox(sample, template, background, textsize)

def sort_boxes(boxes, max_distance=0.3):
	total_numb = len(boxes)

	line_1 = []
	sorted_line_1 = []
	line_2 = []
	sorted_line_2 = []

	min_y = np.min(boxes[:, 1])

	for i in range(total_numb):
		if math.fabs(boxes[i][1]-min_y) < max_distance:
			line_1.append(boxes[i])
		else:
			line_2.append(boxes[i])

	sorted_line_1 = [x for x in sorted(line_1, key = lambda line_1: line_1[0])]
	if len(line_2) > 0:
		sorted_line_2 = [x for x in sorted(line_2, key = lambda line_2: line_2[0])]
	return sorted_line_1 + sorted_line_2

def segment_and_get_boxes(img, sample, textsize, margin = 3):
    total_char = len(sample.replace('.', '').replace('/', '').replace('-', ''))
    if type(textsize[0]).__name__ == 'tuple':
        tmp = list(textsize).copy()
        textsize = list(tmp[0])
        textsize[0] = textsize[0] + tmp[1][0]
    else: 
        textsize = list(textsize)
    height, width, _ = img.shape
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    thresh[:, 0:int(width*0.04)] = 0
    thresh[:, int(width*0.96):] = 0
    thresh[:int(height*0.05), :] = 0
    thresh[int(height*0.95):, :] = 0
    # cv2.imshow('thresh', thresh)
    contours, hier = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
    list_box = []
    for i in range(len(sorted_contours)):
        x, y, w, h = cv2.boundingRect(sorted_contours[i])
        x += random.randint(-margin, 0)
        y += random.randint(-margin, 0)
        w += random.randint(0, 2*margin)
        h += random.randint(0, 2*margin)
        list_box.append([x, y, (x+w), (y+h)])
    list_box = nms_fast(np.array(list_box), overlapThresh = 0.1)[: total_char]
    list_box = format_boundingbox(np.array(list_box), width, height)
    sorted_list_box = sort_boxes(list_box)
    return sorted_list_box


def generate_sample(template):
    count_numb = template.count('N')
    count_char = template.count('C')
    count_all = template.count('*')
    for i in range(count_numb):
        idx_numb = random.randint(0, total_number - 1)
        template = template.replace('N', available_number[idx_numb], 1)
    for i in range(count_char):
        idx_char = random.randint(0, total_char - 1)
        template = template.replace('C', available_char[idx_char], 1)
    for i in range(count_all):
        idx_char = random.randint(0, total_char + total_number - 1)
        template = template.replace('*', available_all[idx_char], 1)
    return template

def generate_plate(template):
	if '/' in template:
		bg = available_square_bg[random.randint(0, len(available_square_bg) - 1)]
		return generate_2lines_image(template, bg)
	else:
		bg = available_rec_bg[random.randint(0, len(available_rec_bg) - 1)]
		return generate_1line_image(template, bg)

def generate_yolo_label(boxes, sample_formated, filename):
	#print(sample_formated)
	assert len(boxes) == len(sample_formated)
	filename_txt = filename.split('.')[0] + '.txt'
	# Delete current label file
	open(filename_txt, 'w+')
	# Write yolo label
	with open(filename_txt, 'a') as f:
		for i in range(len(boxes)):
			x, y, w, h = boxes[i]
			f.write('{} {} {} {} {}\n'.format(box_label[sample_formated[i]], x, y, w, h))


def visualize(img, boxes, label):
	height, width, _ = img.shape
	#print(boxes)
	for i in range(len(label)):
		x, y, w, h = boxes[i]
		x, y, w, h = int(x*width), int(y*height), int(w*width), int(h*height)
		cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 0, 255), 2)
		cv2.putText(img, label[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
	#cv2.imwrite('syn_labeled.jpg', img)
	cv2.imshow('result', img)
	cv2.waitKey(0) 

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Vietnamese Synthesis License Plate.')

	parser.add_argument('--numb', default=1,
	                   help='Total number of Synthesis images')

	parser.add_argument('--output_dir', default='output',
	                   help='Output directory')

	args = parser.parse_args()
	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)
	err = 0
	for i in progressbar.progressbar(range(int(args.numb))):
		try:
			filename = os.path.join(args.output_dir ,'syn_{}.jpg'.format(i))
			idx = random.randint(0, total_template - 1)
			template = available_template[idx]
			sample = generate_sample(template)
			base_img, textsize = generate_plate(sample)
			# aug_img = augmention(base_img)
			width, height = base_img.size
			boxes = segment_and_get_boxes(np.array(base_img), sample, textsize)
			labels = sample.replace('-', '').replace('.', '').replace('/', '')
			#generate_yolo_label(boxes, labels, filename)
			base_img.save(filename)
			if visual:
				visualize(np.array(base_img), boxes, labels)
		except AssertionError:
			err += 1
	print('Completed !')
	print('Error: {} images'.format(err))
	


