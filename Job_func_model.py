import re
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F


vocab_mapping = torch.load('vocab_mapping.pth')
class_mapping = torch.load('class_mapping.pth')
checkpoint = torch.load('checkpoint.pth', map_location='cpu')

class neural_network3(nn.Module):
    def __init__(self, input_size, vocab_size, output_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(input_size*embedding_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, output_size)
        self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        x = self.embedding(x.long())
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x




input_size   = checkpoint['input_size']
vocab_size   = checkpoint['vocab_size']
output_size  = checkpoint['output_size']
embedding_dim= checkpoint['embedding_dim']

inference_model = neural_network3( input_size=input_size,vocab_size=vocab_size, output_size=output_size, embedding_dim=embedding_dim)
inference_model.load_state_dict(checkpoint['state_dict'])


def pad_features(integerInput, seq_length):

    features = np.zeros(seq_length, dtype=int)

    if len(integerInput)<6:
      features[-len(integerInput):] = np.array(integerInput)[:seq_length]
    else:
      features[:] = np.array(integerInput)[:seq_length]

    return features

def process_user_input(input_text):
	input_words = re.sub("[^\w]", " ",  input_text).split()
	input_as_int=[]
	for k,word in enumerate(input_words):
		if word.lower() in vocab_mapping.keys():
			input_as_int.append(vocab_mapping[word.lower()]) 
		else:
			input_as_int.append(0)

	model_input = pad_features(input_as_int, seq_length=6)
	return model_input


def infer( user_input, prob_thresh=0.35 ):

	model_input = process_user_input(user_input)
	inference_model.eval()
	model_output = inference_model(torch.from_numpy(model_input).unsqueeze(0))
	segmoid = nn.Sigmoid()
	probs = segmoid(model_output[0])
	probs_above_thresh = probs>prob_thresh
	output_classes = np.argwhere(probs_above_thresh)
	final_output = [class_mapping[i.item()].capitalize() for i in output_classes[0]]
	return final_output




















