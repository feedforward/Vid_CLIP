#!/usr/bin/env python
# coding: utf-8


# GPU setup
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"


from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import numpy as np
import torch
import clip
from tqdm.notebook import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import random
import os
import numpy as np
from PIL import Image
import torchvision   
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sklearn
from torch import nn
from torch.nn.utils.rnn import *

cuda = torch.cuda.is_available()
print("cuda", cuda)
num_workers = 8 if cuda else 0
print(num_workers)
print("Torch version:", torch.__version__)


# # Load CLIP Model


print("Avaliable Models: ", clip.available_models())
model, preprocess = clip.load("RN50") # clip.load("ViT-B/32") #

input_resolution = model.input_resolution #.item()
context_length = model.context_length #.item()
vocab_size = model.vocab_size #.item()

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)


# # Selected classes and mapping for kinetics dataset


labels = ['making_tea',
		'shaking_head',
		'skiing_slalom',
		'bobsledding',
		'high_kick',
		'scrambling_eggs',
		'bee_keeping',
		'swinging_on_something',
		'washing_hands',
		'laying_bricks',
		'push_up',
		'doing_nails',
		'massaging_legs',
		'using_computer',
		'clapping',
		'drinking_beer',
		'eating_chips',
		'riding_mule',
		'petting_animal_(not_cat)',
		'frying_vegetables',
		'skiing_(not_slalom_or_crosscountry)',
		'snowkiting',
		'massaging_person\'s_head',
		'cutting_nails',
		'picking_fruit']

map_id = {}
i=0
for label in labels:
	map_id[label]=i
	i+=1

map_id


import os
import pdb
ROOT = "/data2/puppala/data/kinetics_jpg/validation"
DEST = "/data2/puppala/data/kinetics_embeddings/validation"
cnt = 1
for filename in labels:
	if filename not in os.listdir(ROOT):
		print(filename," - Missing")


import os
import pdb
ROOT = "/data2/puppala/data/kinetics_jpg/validation"
DEST = "/data2/puppala/data/kinetics_embeddings/validation"
cnt = 1
class_names = os.listdir(ROOT)
len_classes = len(labels)
for cls_idx, filename in enumerate(labels):
	class_file = os.path.join(ROOT,filename)
	random_tags = os.listdir(class_file)
	len_random_tags = len(random_tags)
	for rand_id, random_tag in enumerate(random_tags):
		try:
			video_name = os.path.join(class_file,random_tag)
			# filename = filename.replace("_"," ")
			count = 0
			frames = sorted([ x for x in os.listdir(video_name) if x.endswith('.jpg')])
			N = len(frames)
			n = N//100
			selected_frames = np.arange(0,N,n).tolist()[0:100]
			for frame in frames:
				if count in selected_frames:
					tmp = str(count)
					image = Image.open(os.path.join(video_name,frame))
					image = preprocess(image)
					image = torch.unsqueeze(image, 0)
					image = image.cuda()
					image_features = model.encode_image(image)
					image_features /= image_features.norm(dim=-1, keepdim=True)
					image_features = image_features.detach().cpu().numpy()
					if count==0:
						a = image_features
					else:
						a = np.vstack((a,image_features))
				count += 1
			cnt+=1

			label = map_id[filename]*np.ones(a.shape[0])
			file_save = os.path.join(DEST,str(cnt))
			np.savez(file_save, data=a,label=label)
			print(cls_idx, '/', len_classes, '-', rand_id, '/', len_random_tags, end='\r')
		except:
			pass
	cnt+=1    

