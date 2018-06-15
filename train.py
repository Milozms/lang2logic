import torch
from model import Model
import utils
from utils import Dataset
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import logging
import os
import json


if __name__ == '__main__':
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	handler = logging.FileHandler("./log/log.txt", mode='w')
	handler.setLevel(logging.INFO)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
	handler.setFormatter(formatter)
	console.setFormatter(formatter)
	logger.addHandler(handler)
	logger.addHandler(console)

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='data')
	parser.add_argument('--vocab_dir', type=str, default='vocab')
	parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
	parser.add_argument('--hidden', type=int, default=300, help='RNN hidden state size.')
	parser.add_argument('--num_layers', type=int, default=1, help='Num of RNN layers.')
	parser.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
	parser.add_argument('--device', type=str, default="cuda:0", help='Device')

	parser.add_argument('--lr', type=float, default=0.01, help='Applies to SGD and Adagrad.')
	parser.add_argument('--lr_decay', type=float, default=0.95)

	parser.add_argument('--num_epoch', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=20)
	parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
	args = vars(parser.parse_args())

	with open(args['vocab_dir'] + '/vocab_logic.json', 'r') as f:
		logic_vocab = json.load(f)
	logic_vocab = utils.vocab_prefix + logic_vocab
	logic2id = {}
	for idx, word in enumerate(logic_vocab):
		logic2id[word] = idx

	with open(args['vocab_dir'] + '/vocab_lang.json', 'r') as f:
		word_vocab = json.load(f)
	word_vocab = utils.vocab_prefix + word_vocab
	with open(args['vocab_dir'] + '/word2id.json', 'r') as f:
		word2id = json.load(f)
	with open(args['vocab_dir'] + '/emb.pkl', 'rb') as f:
		emb_mat = pickle.load(f)

	assert emb_mat.shape[0] == len(word_vocab)
	assert emb_mat.shape[1] == args['emb_dim']
	args['vocab_size'] = len(word_vocab)
	args['out_vocab_size'] = len(logic_vocab)
	niter = args['num_epoch']

	device = torch.device("%s" % args['device'] if torch.cuda.is_available() else "cpu")

	train_dset = Dataset(1, 'train', args, word2id, logic2id, device, remain_size=100, shuffle=True)
	# dev_dset = Dataset(1, 'valid', args, word2id, logic2id, device, shuffle=False)


	model = Model(args, device, emb_mat)
	print('Using device: %s' % device.type)

	# model.eval(dev_dset)

	# Training
	min_loss = 0.0
	for iter in range(niter):
		print('Iteration %d:' % iter)
		loss = 0.0
		for idx, batch in enumerate(tqdm(train_dset.batched_data)):
			ques, logic = batch
			loss_batch = model.train_batch(ques, logic)
			loss += loss_batch
		loss /= len(train_dset.batched_data)
		print('Loss: %f' % loss)

		# valid_loss, decoded = model.eval(dev_dset)
		# utils.output_to_file(decoded, logic_vocab, './output/output%d.txt' % iter)
		# print('\n')
		# if valid_loss < min_loss:
		# 	min_loss = valid_loss
		# 	model.save('./save_model/model', iter)
		# logging.info('Iteration %d, Train loss %f, Valid loss %f' % (iter, loss, valid_loss))
		logging.info('Iteration %d, Train loss %f' % (iter, loss))

