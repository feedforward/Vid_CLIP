from torch.utils.data import Dataset
import os
import torch
import numpy as np


# Dataset class for Train and Dev
class Mapped_Dataset(Dataset):
	def __init__(self, path, class_names = None):
		self.path = path
		if class_names is None:
			self.class_names = os.listdir(self.path)
		else:
			self.class_names = class_names
		
		self.data = []
		for class_name in self.class_names:
			class_path = os.path.join(self.path, class_name)
			files = [ x for x in os.listdir(class_path) 
					 if (x.endswith('.npz') and not x.endswith('_preattention_features.npz'))]
			files = [ os.path.join(class_path, x) for x in files]
			self.data += files


	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, index):
		data = np.load(self.data[index] ,allow_pickle=True)
		sample = torch.from_numpy(data['data']).type(torch.FloatTensor)
		label  = torch.tensor(data['label'])#.type(torch.LongTensor)
		length = torch.tensor(100)
		return sample,label,length
    
    
# Dataset class for Train and Dev
class Unmapped_Dataset(Dataset):
	def __init__(self, path):
		self.path = path
		self.data = []
		for file in os.listdir(self.path):
			if fnmatch.fnmatch(file, '*.npz'):
				self.data.append(file)


	def __len__(self):
		return len(self.data)
	def __getitem__(self, index):
		data = np.load(os.path.join(self.path,self.data[index]),allow_pickle=True)
		sample = torch.from_numpy(data['data']).type(torch.FloatTensor)
		label  = torch.tensor(data['label'])#.type(torch.LongTensor)
		length = torch.tensor(100)
		return sample,label,length