from .constants import *
from .models import *
from .utils import *

import os
from sys import argv
from time import time
import torch.nn.functional as F
from torch.distributions import Categorical


class Agent:
    def __init__(self, agent_id, model, optimizer):
        self.agent_id = agent_id
        self.model = model
        self.optimizer = optimizer


def backprop(agent_id, schedule_t, value_t, schedule_next, optimizer, model):
    optimizer[agent_id].zero_grad()
    value, action = model[agent_id](schedule_t)
    loss = criterion(value, value_t) + criterion(action, schedule_next)
    loss.backward()
    optimizer[agent_id].step()
    vl = loss.detach().item()
    pl = (action.argmax(dim=1) != schedule_next.argmax(dim=1)).sum().item()
    return vl, pl, action


def save_model(agent, epoch, accuracy_list):
    model = agent.model
    optimizer = agent.optimizer

    file_path = MODEL_SAVE_PATH + "/" + model.name + "_" + str(epoch) + ".ckpt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy_list': accuracy_list
    }, file_path)

def load_model(filename, model, data_type):
	optimizer = torch.optim.AdamW(model.parameters() , lr=0.0001, weight_decay=1e-5)
	file_path = MODEL_SAVE_PATH + "/" + model.name + "_Trained.ckpt"
	if os.path.exists(file_path):
		print(color.GREEN+"Loading pre-trained model: "+filename+color.ENDC)
		checkpoint = torch.load(file_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		epoch = -1; accuracy_list = []
		print(color.GREEN+"Creating new model: "+model.name+color.ENDC)
	return model, optimizer, epoch, accuracy_list