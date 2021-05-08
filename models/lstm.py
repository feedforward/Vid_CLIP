import pdb
import torch.nn as nn
import torch

# LSTM Model
NUM_CLASSES = 25
class LSTM_Model(nn.Module):
	def __init__(self, input_feature_size, embed_size, hidden_size, zeroshot_weights):
		super(Model, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv1d(input_feature_size, embed_size , kernel_size=2),
			nn.ReLU(),
		)
		# No of layers ---> reduce
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=3,bidirectional=True)
		self.output1 = nn.Linear(hidden_size * 2, 1024)
		# self.output2 = nn.Linear(1024,1024)
		self.dummy = nn.Linear(1024,NUM_CLASSES,bias=False)
		self.dummy.weight = torch.nn.Parameter(zeroshot_weights.float().T.clone(), requires_grad=False)

		# self.dummy.requires_grad = True
		# self.output3 = zeroshot_weights_.float().cuda()


	def forward(self, X, lengths):
		X_ = torch.transpose(X,2,1)
		X_ = F.pad(input=X_, pad=(0,1,0,0), mode='constant', value=0)
		X = self.layer1(X_)
		X = torch.transpose(X,0,2)
		X = torch.transpose(X,1,2)
		packed_X = pack_padded_sequence(X, lengths.cpu(), enforce_sorted=False)
		packed_out,(h_n,c_n) = self.lstm(packed_X)
		out,_ = pad_packed_sequence(packed_out)
		out = self.output1(out[0,:,:]) # 1024
		# out /= out.norm(dim=-1, keepdim=True) # Normalize the logits. #### SHOULD WE MULTIPLY BY 100
		logits = self.dummy(out)
		return logits

def init_weights(m):
	if type(m) == nn.Conv1d or type(m) == nn.Linear:
		torch.nn.init.xavier_normal_(m.weight.data)