import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
from torch.utils import data
from torch import optim
import torch.utils.data
import argparse
import warnings
import sys
from models import Encoder
from data_provision_tools import \
	HDF5Dataset, \
	BalancedMultimodalSampler, \
	ImbalancedMutlimodalSampler, \
	BalancedBinarySampler, \
	ImbalancedBinarySampler, \
	DatasetBag

warnings.filterwarnings("ignore")

argv = sys.argv[1:]

print("CUDA current device " + str(torch.cuda.current_device()))
print("CUDA devices available " + str(torch.cuda.device_count()))


# parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment', type=int, default=0)
parser.add_argument('-c', '--CNN', help='cnn_to_use', type=str, default='resnet34')
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size', type=int, default=256)
parser.add_argument('-e', '--EPOCHS', help='epochs to train', type=int, default=10)
parser.add_argument(
	'-f', '--features', help='features_to_use: embedding (True) or features from CNN (False)',
	type=str, default='True'
)
parser.add_argument('-q', '--MOCO_QUEUE', help='queue size for MoCo algorithm', type=int, default=8192)
parser.add_argument('-l', '--lr', help='learning rate', type=float, default=1e-4)
parser.add_argument(
	'-p', '--BASE_PATH', help='base path', type=str, default='/projects/0/examode/MIT_PP3/MoCo_representation_training'
)
parser.add_argument(
	'-i', '--H5_INPUT', help='input h5 file name', type=str, default='sampled_dataset.h5'
)
parser.add_argument(
	'-a', '--APPROACH', help='approach to class labels - binary (pubmed vs not-pubmed) or mutliclass',
	type=str, choices=['binary', 'multiclass'], default='binary'
)
parser.add_argument(
	'-o', '--output_folder', help='path where to store the model weights', type=str, default='models/'
)


args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
EPOCHS = args.EPOCHS
EPOCHS_str = str(EPOCHS)
EMBEDDING_bool = args.features
lr = args.lr
APPROACH = args.APPROACH
num_keys = args.MOCO_QUEUE

base_path = args.BASE_PATH
OUTPUT_FOLDER = args.output_folder
OUTPUT_FOLDER = os.path.join(base_path, OUTPUT_FOLDER)
input_filename = args.H5_INPUT
input_path = os.path.join(base_path, input_filename)
dataloader_num_workers = 4

if EMBEDDING_bool == 'True':
	EMBEDDING_bool = True
else:
	EMBEDDING_bool = False

seed = N_EXP
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

print("PARAMETERS")
print("N_EPOCHS: " + str(EPOCHS_str))
print("CNN used: " + str(CNN_TO_USE))
print("BATCH_SIZE: " + str(BATCH_SIZE_str))
print("APPROACH: " + APPROACH)


# create folder (used for saving weights)
def create_dir(path_):
	if not os.path.isdir(path_):
		try:
			os.mkdir(path_)
		except OSError:
			print("Creation of the directory %s failed" % path_)
		else:
			print("Successfully created the directory %s " % path_)


print("CREATING/CHECKING DIRECTORIES")

create_dir(OUTPUT_FOLDER)

models_path = OUTPUT_FOLDER
# path model file
model_weights_filename = os.path.join(models_path, f"PP3_MoCo_{APPROACH}.pt")
model_weights_filename_temporary = os.path.join(models_path, f"PP3_MoCo_{APPROACH}_temporary.pt")

pre_trained_network = torch.hub.load('pytorch/vision:v0.10.0', CNN_TO_USE, pretrained=True)

if ('resnet' in CNN_TO_USE) or ('resnext' in CNN_TO_USE):
	fc_input_features = pre_trained_network.fc.in_features
elif 'densenet' in CNN_TO_USE:
	fc_input_features = pre_trained_network.classifier.in_features
elif 'mobilenet' in CNN_TO_USE:
	fc_input_features = pre_trained_network.classifier[1].in_features


# moco_dim = 768
moco_dim = 512
moco_m = 0.999
temperature = 0.07

batch_size = BATCH_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_domains = 1 if APPROACH == "binary" else 3
print(f"n_domains: {n_domains}")

encoder = Encoder(dim=moco_dim, cnn_to_use=CNN_TO_USE, n_domains=n_domains).to(device)
momentum_encoder = Encoder(dim=moco_dim, cnn_to_use=CNN_TO_USE, n_domains=n_domains).to(device)

encoder.embedding.weight.data.normal_(mean=0.0, std=0.01)
encoder.embedding.bias.data.zero_()

momentum_encoder.load_state_dict(encoder.state_dict(), strict=False)

for param in momentum_encoder.parameters():
	param.requires_grad = False

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in encoder.parameters())
print(f'{total_params:,} total parameters.')

total_trainable_params = sum(
	p.numel() for p in encoder.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

torch.backends.cudnn.benchmark = True


def loss_function(q, k, queue):

	# N is the batch size
	N = q.shape[0]
	
	# C is the dimension of the representation
	C = q.shape[1]

	# BMM stands for batch matrix multiplication
	# If mat1 is B × n × M tensor, then mat2 is B × m × P tensor,
	# Then output a B × n × P tensor.
	pos = torch.exp(torch.div(torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).view(N, 1), temperature))
	
	# Matrix multiplication is performed between the query and the queue tensor
	neg = torch.unsqueeze(
		torch.sum(torch.exp(torch.div(torch.mm(q.view(N, C), torch.t(queue)), temperature)), dim=1),
		dim=1
	)

	# Sum up
	denominator = neg + pos

	return torch.mean(-torch.log(torch.div(pos, denominator)))


criterion = torch.nn.CrossEntropyLoss().to(device)

# since here we have domains one-hot encoded and we actually have domain classifier, we will use:
if APPROACH == 'binary':
	criterion_domain = torch.nn.BCELoss().to(device)
elif APPROACH == 'multiclass':
	criterion_domain = torch.nn.CrossEntropyLoss().to(device)

lambda_val = 0.5

optimizer_str = 'adam'

# Optimizer
SGD_momentum = 0.9
weight_decay = 1e-4
shuffle_bn = True

if optimizer_str == 'sgd':
	optimizer = optim.SGD(encoder.parameters(), lr=lr, momentum=SGD_momentum, weight_decay=weight_decay)

elif optimizer_str == 'adam':
	optimizer = optim.Adam(encoder.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=True)

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


def momentum_step(m=1):
	"""
	Momentum step (Eq (2)).
	Args:
		- m (float): momentum value. 1) m = 0 -> copy parameter of encoder to key encoder
									 2) m = 0.999 -> momentum update of key encoder
	"""
	params_q = encoder.state_dict()
	params_k = momentum_encoder.state_dict()
	
	dict_params_k = dict(params_k)
	
	for name in params_q:
		theta_k = dict_params_k[name]
		theta_q = params_q[name].data
		dict_params_k[name].data.copy_(m * theta_k + (1-m) * theta_q)

	momentum_encoder.load_state_dict(dict_params_k)


def update_lr(epoch):
	"""
	Learning rate scheduling.
	Args:
		- epoch (float): Set new learning rate by a given epoch.
	"""
	if epoch < 10:
		lr = args.lr
	elif epoch >= 20:
		lr = args.lr * 0.01
	else:  # (epoch >= 10) and (epoch < 20)
		lr = args.lr * 0.1
	
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def update_queue(queue, k):
	new_queue = torch.cat([k, queue], dim=0)
	new_queue = new_queue[:num_keys]
	return new_queue


''' ######################## < Step 4 > Start training ######################## '''

# Initialize momentum_encoder with parameters of encoder.
momentum_step(m=0)

# Dataset initialization
training_dataset = HDF5Dataset(input_path, "init", approach=APPROACH)
# classes = np.stack([entry[2] for entry in training_dataset])
classes = training_dataset._hf[training_dataset.domain_gt_dataset_name][()]

if APPROACH == "binary":
	sampler = BalancedBinarySampler(classes)
elif APPROACH == "multiclass":
	sampler = BalancedMultimodalSampler(classes)

dataloader_params = {
	'batch_size': batch_size,
	'num_workers': dataloader_num_workers,
	'sampler': sampler
}
# dataloader_params = {
# 	'batch_size': batch_size,
# 	'num_workers': dataloader_num_workers,
# 	'shuffle': True
# }
training_dataloader = data.DataLoader(training_dataset, **dataloader_params)
training_iterator = iter(training_dataloader)

# Training
print('\nStart training!')

# iterations_per_epoch = 8600
iterations_per_epoch = int(len(training_dataset) / training_dataloader.batch_size)

losses_train = []

# number of epochs without improvement
EARLY_STOP_NUM = 10
early_stop_cont = 0
num_epochs = EPOCHS
best_loss = 100000.0

tot_iterations = num_epochs * iterations_per_epoch
cont_iterations_tot = 0

grl_alpha_starting_point = 0.1

TEMPERATURE = 0.07

epoch = 0

while epoch < num_epochs and early_stop_cont < EARLY_STOP_NUM:
	print(f"Starting epoch {epoch}")
	total_iters = 0
	
	# accumulator loss for the outputs
	train_loss = 0.0
	train_loss_domain = 0.0
	train_loss_moco = 0.0
	
	# if loss function lower
	is_best = False

	print(f"\nInitializing a queue with {num_keys} keys.")

	queue = torch.Tensor().to(device)

	# change data processing mode, see H5Dataset
	training_iterator._dataset.mode = "queue_building"
	while queue.shape[0] < num_keys:
		try:
			_, img, _ = next(training_iterator)
		except StopIteration:
			training_iterator = iter(training_dataloader)
			_, img, _ = next(training_iterator)
		with torch.no_grad():
			key_feature = momentum_encoder(img.to(device), 'valid', None)
			queue = torch.cat([queue, key_feature])

	training_iterator._dataset.mode = "train"
	encoder.train()
	momentum_encoder.train()

	for i in range(iterations_per_epoch):
		print(f"[epoch {epoch: >4}]: iteration {i+1: >5} of {iterations_per_epoch}")

		try:
			x_q, x_k, domain_oh = next(training_iterator)
		except StopIteration:
			training_iterator = iter(training_dataloader)
			x_q, x_k, domain_oh = next(training_iterator)

		p = float(cont_iterations_tot + epoch * tot_iterations) / num_epochs / tot_iterations

		alpha = 2. / (1. + np.exp(-10 * p)) - 1
		alpha = alpha + grl_alpha_starting_point*(1 - alpha)

		# Preprocess
		# momentum_encoder.train()
		# momentum_encoder.zero_grad()
		# encoder.train()
		# encoder.zero_grad()

		# Shuffled BN : shuffle x_k before distributing it among GPUs (Section. 3.3)
		if shuffle_bn:
			idx = torch.randperm(x_k.size(0))
			x_k = x_k[idx]

		# x_q, x_k : (N, 3, 64, 64)
		x_q, x_k, domain_oh = x_q.to(device), x_k.to(device), domain_oh.type(torch.FloatTensor).to(device)

		q, he_q = encoder(x_q, "train", alpha)  # q : (N, 128)
		with torch.no_grad():
			k = momentum_encoder(x_k, 'valid', _).detach()  # k : (N, 128)

		# Shuffled BN : unshuffle k (Section. 3.3)
		if shuffle_bn:
			k_temp = torch.zeros_like(k)
			for a, j in enumerate(idx):
				k_temp[j] = k[a]
			k = k_temp
		"""
		# positive logits: Nx1
		l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
		# negative logits: NxK
		l_neg = torch.einsum('nc,ck->nk', [q, queue.t()])

		# Positive sampling q & k
		#l_pos = torch.sum(q * k, dim=1, keepdim=True) # (N, 1)
		#print("l_pos",l_pos)

		# Negative sampling q & queue
		#l_neg = torch.mm(q, queue.t()) # (N, 4096)
		#print("l_neg",l_neg)

		# Logit and label
		logits = torch.cat([l_pos, l_neg], dim=1) / temperature # (N, 4097) witi label [0, 0, ..., 0]
		labels = torch.zeros(logits.size(0), dtype=torch.long).to(device)

		# Get loss and backprop
		loss_moco = criterion(logits, labels)
		"""
		loss_moco = loss_function(q, k, queue)
		loss_domains = lambda_val * criterion_domain(he_q, domain_oh)

		loss = loss_moco + loss_domains
		loss.backward()

		# Encoder update
		optimizer.step()

		momentum_encoder.zero_grad()
		encoder.zero_grad()

		# Momentum encoder update
		momentum_step(m=moco_m)

		# Update dictionary
		queue = update_queue(queue, k)

		# Print a training status, save a loss value, and plot a loss graph.

		train_loss_moco = train_loss_moco + ((1 / (total_iters+1)) * (loss_moco.item() - train_loss_moco))
		train_loss_domain = train_loss_domain + ((1 / (total_iters+1)) * (loss_domains.item() - train_loss_domain))
		total_iters += 1
		cont_iterations_tot = cont_iterations_tot + 1
		train_loss = train_loss_moco + train_loss_domain

		print('[Epoch : %d / Total iters : %d] : loss_moco :%f, loss_domain :%f ...' %(epoch, total_iters, train_loss_moco, train_loss_domain))
			
		if i % 10 == 0:
			print('a')
			if best_loss > train_loss_moco:
				early_stop_cont = 0
				print("=> Saving a new best model")
				print("previous loss : " + str(best_loss) + ", new loss function: " + str(train_loss_moco))
				best_loss = train_loss_moco
				try:
					torch.save(encoder.state_dict(), model_weights_filename,_use_new_zipfile_serialization=False)
				except:
					torch.save(encoder.state_dict(), model_weights_filename)
			else:
				early_stop_cont += 1
				try:
					torch.save(encoder.state_dict(), model_weights_filename_temporary, _use_new_zipfile_serialization=False)
				except:
					torch.save(encoder.state_dict(), model_weights_filename_temporary)

			torch.cuda.empty_cache()
		
		# Update learning rate
	#update_lr(epoch)

	print("epoch "+str(epoch) + " train loss: " + str(train_loss))

	epoch += 1
	if early_stop_cont == EARLY_STOP_NUM:
		print("EARLY STOPPING")
