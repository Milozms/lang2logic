import tensorflow as tf
from utils import Dataset
from tqdm import tqdm
import logging
import pickle
import numpy as np
import json
import os
import math
from tensorflow.contrib import crf


def restore_tokens(out_idx, vocab):
	out_tokens = []
	for idx in out_idx:
		if idx <= 2:
			break
		else:
			out_tokens.append(vocab[idx])
	return out_tokens

def get_gru_cell(hidden, num_layers):
	if num_layers == 1:
		return tf.nn.rnn_cell.GRUCell(num_units=hidden)
	mcell = []
	for i in range(num_layers):
		mcell.append(tf.nn.rnn_cell.GRUCell(num_units=hidden))
	return tf.nn.rnn_cell.MultiRNNCell(mcell)

class Model(object):
	def __init__(self, config, word_emb_mat):
		self.hidden = config.hidden
		self.input_vocab_size = config.input_vocab_size
		self.target_vocab_size = config.target_vocab_size
		self.emb_dim = config.emb_dim
		self.batch = config.batch
		self.is_train = config.is_train
		self.maxlen = config.maxlen
		self.word_emb_mat = word_emb_mat
		self.epoch_num = config.epoch_num
		self.max_grad_norm = config.max_grad_norm
		self.lr = config.lr
		self.num_layers = config.num_layers
		self.maxacc = 0.0
		self.minloss = 100
		self.build()


	def build(self):
		self.input = tf.placeholder(dtype = tf.int32, shape = [None, self.maxlen], name = 'input')
		self.input_len = tf.placeholder(dtype = tf.int32, shape = [None], name = 'input_len')
		self.target = tf.placeholder(dtype = tf.int32, shape = [None, self.maxlen], name = 'target')
		self.target_len = tf.placeholder(dtype = tf.int32, shape = [None], name = 'target_len')
		self.keep_prob = tf.placeholder(dtype = tf.float32, shape = ())
		batch_size = tf.shape(self.input)[0]
		# batch_size = self.input.shape[0].value

		hidden = self.hidden
		target_vocab_size = self.target_vocab_size
		emb_dim = self.emb_dim
		maxlen = self.maxlen

		with tf.variable_scope("embeddings"):
			self.input_embeddings = tf.get_variable(name='input_word_embedding',
													dtype=tf.float32,
													initializer=tf.constant(self.word_emb_mat, dtype=tf.float32),
													trainable=True
													)
			self.target_embeddings = tf.get_variable(name = "target_vocab_embedding",
													 shape=[target_vocab_size, emb_dim],
									dtype = tf.float32,
									initializer = tf.random_normal_initializer(),
									trainable=True)

		input_emb = tf.nn.embedding_lookup(self.input_embeddings, self.input)
		target_emb = tf.nn.embedding_lookup(self.target_embeddings, self.target)

		with tf.variable_scope("encoder"):
			encoder_cell = get_gru_cell(hidden, self.num_layers)
			init_state = encoder_cell.zero_state(batch_size, dtype=tf.float32)
			encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, input_emb, self.input_len, init_state)

		# attention
		def attention(query, step_i):
			with tf.variable_scope("attention") as att_scope:
				if step_i != 0:
					att_scope.reuse_variables()
				query_ = tf.expand_dims(query, axis=2)  # [batch, hidden, 1]
				attn = tf.matmul(encoder_outputs, query_)  # [batch, maxlen, 1]
				attn = tf.squeeze(attn, axis=2)		# [batch, maxlen]
				masks = tf.sequence_mask(self.input_len, self.maxlen, dtype=tf.float32)
				attn_w = tf.nn.softmax(attn, dim=1)
				attn_weight = tf.multiply(attn_w, masks)		# [batch, maxlen]
				attn_weight = tf.expand_dims(attn_weight, axis=1)		# [batch, 1, maxlen]
				c = tf.squeeze(tf.matmul(attn_weight, encoder_outputs), axis=1)
			return c

		self.decoder_cell = get_gru_cell(hidden, self.num_layers)
		# self.decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob=self.keep_prob)
		dec_targets = tf.unstack(target_emb, axis=1)
		prev_out = None
		out_idx = []
		loss_steps = []
		# pred_size = target_vocab_size + 1  #??????
		label_steps = tf.unstack(self.target, axis=1)
		# initial state
		prev_hidden = encoder_state
		# prev_hidden = decoder_cell.zero_state(batch_size, dtype=tf.float32)
		sos = tf.ones(shape=[batch_size], dtype=tf.int32)
		sos_emb = tf.nn.embedding_lookup(self.target_embeddings, sos)
		with tf.variable_scope("decoder") as decoder_scope:
			outputs = []
			for time_step in range(maxlen):
				if time_step >= 1:
					decoder_scope.reuse_variables()
				if time_step == 0:
					cur_in = sos_emb # <SOS>
				else:
					if self.is_train:
						cur_in = dec_targets[time_step - 1] # [batch, word_dim]
					else:
						cur_in = prev_out # [batch, word_dim]
				cell_in = cur_in
				cur_out, cur_hidden = self.decoder_cell(cell_in, prev_hidden) # [batch, hidden]
				prev_hidden = cur_hidden

				# attention
				c = attention(cur_out, time_step)
				output_w1 = tf.get_variable('output_w1', shape=[hidden, hidden],
										   initializer=tf.random_normal_initializer())
				output_w2 = tf.get_variable('output_w2', shape=[hidden, hidden],
											initializer=tf.random_normal_initializer())
				# h_att = tf.tanh(tf.matmul(cur_out, output_w1) + tf.matmul(c, output_w2))
				h_att = tf.concat([cur_out, c], axis=1)
				# output_w3 = tf.get_variable('output_w3', shape=[hidden*2, hidden],
				# 							initializer=tf.random_normal_initializer())
				# h_att = tf.tanh(tf.matmul(h_att, output_w3))
				# output projection to logic tokens
				output_w = tf.get_variable('output_w', shape=[hidden*2, target_vocab_size],
										   initializer = tf.random_normal_initializer())
				output_b = tf.get_variable('output_b', shape=[target_vocab_size],
										   initializer=tf.random_normal_initializer())
				output = tf.matmul(h_att, output_w) + output_b # [batch, pred_size]
				output_softmax = tf.nn.softmax(output, dim=1)
				outputs.append(output_softmax)

				if self.is_train:
					labels = label_steps[time_step]
					loss_steps.append(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = output))

				out_index = tf.argmax(output, 1) # [batch, vocab_size]
				out_idx.append(out_index)
				# input for next cell
				prev_out = tf.nn.embedding_lookup(self.target_embeddings, out_index) # [batch, word_emb_dim]

		out_idx = tf.transpose(tf.stack(out_idx)) # [batch_size, timesteps]

		if self.is_train == False:
			self.out = out_idx
			return

		loss = tf.transpose(tf.stack(loss_steps)) # [batch_size, maxlen - 1]
		# mask loss
		loss_mask = tf.sequence_mask(self.target_len, maxlen, tf.float32)
		loss = tf.reduce_mean(loss_mask * loss)

		# self.out_test = [loss, out_idx]

		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		# optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.95)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.max_grad_norm)
		train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())
		self.out = [loss, train_op, out_idx]
		self.out_valid = [loss, out_idx]
		return


	def decode_test_model(self, sess, test_dset, niter, logic_vocab, saver, dir):
		'''
		greedy search
		'''
		test_dset.current_index = 0
		num_batch = int(math.ceil(test_dset.datasize / self.batch))
		out_idx = []
		outf = open(dir + '/output' + str(niter) + '.txt', 'w')
		acc_cnt = 0.0
		all_cnt = 0.0
		for bi in tqdm(range(num_batch)):
			mini_batch = test_dset.batched_data[bi]
			questions, logics, ques_lens, logic_lens = mini_batch
			feed_dict = {}
			feed_dict[self.input] = questions
			# feed_dict[self.target] = logics
			feed_dict[self.input_len] = ques_lens
			# feed_dict[self.target_len] = logic_lens
			feed_dict[self.keep_prob] = 1.0
			out_idx_cur = sess.run(self.out, feed_dict=feed_dict)
			out_idx_cur = np.array(out_idx_cur, dtype=np.int32)
			out_idx_lst = [list(x) for x in out_idx_cur]
			out_idx += out_idx_lst
			for i in range(len(questions)):
				output_tokens = restore_tokens(out_idx_cur[i], logic_vocab)
				golden_tokens = restore_tokens(logics[i], logic_vocab)
				logic = '\t'.join(output_tokens)
				golden = '\t'.join(golden_tokens)
				outf.write(logic + '\n' + golden + '\n\n')
				if output_tokens == golden_tokens:
					acc_cnt += 1.0
				all_cnt += 1.0
			acc = acc_cnt/all_cnt
		logging.info('Iter %d, acc = %f' % (niter, acc))
		if acc > self.maxacc:
			self.maxacc = acc
			saver.save(sess, './savemodel/model' + str(niter) + '.pkl')
		outf.close()


	def valid_model(self, sess, valid_dset, niter, saver):
		valid_dset.current_index = 0
		num_batch = int(math.ceil(valid_dset.datasize / self.batch))
		out_idx = []
		loss_iter = 0.0
		for bi in tqdm(range(num_batch)):
			mini_batch = valid_dset.batched_data[bi]
			questions, logics, ques_lens, logic_lens = mini_batch
			feed_dict = {}
			feed_dict[self.input] = questions
			feed_dict[self.target] = logics
			feed_dict[self.input_len] = ques_lens
			feed_dict[self.target_len] = logic_lens
			feed_dict[self.keep_prob] = 1.0
			loss, out_idx_cur = sess.run(self.out_valid, feed_dict=feed_dict)
			loss_iter += loss
		loss_iter /= num_batch
		logging.info('iter %d, valid loss = %f' % (niter, loss_iter))
		if loss_iter < self.minloss:
			self.minloss = loss_iter
			saver.save(sess, './savemodel/model'+str(niter)+'.pkl')




