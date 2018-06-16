import tensorflow as tf
from utils import Dataset
from tqdm import tqdm
import logging
import pickle
import numpy as np
import json
import os
import math
from model import Model
import utils

def train(config, train_dset, valid_dset, test_dset, logic_vocab, wordemb):
	with tf.variable_scope('model'):
		model = Model(config, word_emb_mat=wordemb)
	config.is_train = False
	with tf.variable_scope('model', reuse=True):
		mtest = Model(config, word_emb_mat=wordemb)

	saver = tf.train.Saver()
	tfconfig = tf.ConfigProto()
	# tfconfig.gpu_options.allow_growth = True
	sess = tf.Session(config=tfconfig)
	# writer = tf.summary.FileWriter('./graph', sess.graph)
	sess.run(tf.global_variables_initializer())
	num_batch = int(math.ceil(train_dset.datasize / model.batch))
	for ei in range(model.epoch_num):
		train_dset.current_index = 0
		loss_iter = 0.0
		for bi in tqdm(range(num_batch)):
			mini_batch = train_dset.batched_data[bi]
			questions, logics, ques_lens, logic_lens = mini_batch
			feed_dict = {}
			feed_dict[model.input] = questions
			feed_dict[model.target] = logics
			feed_dict[model.input_len] = ques_lens
			feed_dict[model.target_len] = logic_lens
			feed_dict[model.keep_prob] = 1.0
			loss, train_op, out_idx = sess.run(model.out, feed_dict=feed_dict)
			# writer.add_graph(sess.graph)
			loss_iter += loss
		loss_iter /= num_batch
		logging.info('iter %d, train loss: %f' % (ei, loss_iter))
		model.valid_model(sess, valid_dset, ei, saver)
		mtest.decode_test_model(sess, valid_dset, ei, logic_vocab, saver, dir='./output_valid')
		# mtest.decode_test_model(sess, test_dset, ei, wordlist, kblist, saver, dir='./output_test')

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
	flags = tf.flags
	flags.DEFINE_integer('hidden', 300, "")
	flags.DEFINE_integer('emb_dim', 300, "")
	flags.DEFINE_integer('maxlen', 40, "")
	flags.DEFINE_integer('batch', 20, "")
	flags.DEFINE_integer('epoch_num', 30, "")
	flags.DEFINE_boolean('is_train', True, "")
	flags.DEFINE_float('max_grad_norm', 5.0, "")
	flags.DEFINE_integer('num_layers', 1, "")
	flags.DEFINE_float('lr', 0.01, "")
	
	vocab_dir = './vocab'
	with open(vocab_dir + '/vocab_logic.json', 'r') as f:
		logic_vocab = json.load(f)
	logic_vocab = utils.vocab_prefix + logic_vocab
	logic2id = {}
	for idx, word in enumerate(logic_vocab):
		logic2id[word] = idx

	with open(vocab_dir + '/vocab_lang.json', 'r') as f:
		word_vocab = json.load(f)
	word_vocab = utils.vocab_prefix + word_vocab
	with open(vocab_dir + '/word2id.json', 'r') as f:
		word2id = json.load(f)
	with open(vocab_dir + '/emb.pkl', 'rb') as f:
		emb_mat = pickle.load(f)

	flags.DEFINE_integer('input_vocab_size', len(word_vocab), "")
	flags.DEFINE_integer('target_vocab_size', len(logic_vocab), "")

	config = flags.FLAGS

	assert emb_mat.shape[0] == len(word_vocab)
	assert emb_mat.shape[1] == config.emb_dim
	train_dset = Dataset(1, 'train', config, word2id, logic2id, shuffle=True)
	dev_dset = Dataset(1, 'valid', config, word2id, logic2id, shuffle=False)
	train(config, train_dset, dev_dset, dev_dset, logic_vocab, emb_mat)
