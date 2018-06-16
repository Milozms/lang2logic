import numpy as np
import prepro
import random
import re

PAD_ID = 0
MAXLEN = 50 	# 23, 39
SOS_token = 1
EOS_token = 2
vocab_prefix = ['<PAD>', '<SOS>', '<EOS>']

class D1base(object):
	def __init__(self):
		self.read_d1_base()

	def read_d1_base(self):
		self.all_city = set()
		self.city2state = {}
		self.all_state = set()
		self.all_river = set()
		self.all_mount = set()
		self.all_country = set(['us', 'usa', 'china'])
		filename = './base/d1_base'
		with open(filename, 'r') as f:
			for idx, line in enumerate(f.readlines()):
				if idx < 22:
					continue
				line = line.strip()
				tokens = re.split('[\(,]', line)
				if tokens[0] == 'city':
					self.all_city.add(tokens[3].strip('\''))
					self.city2state[tokens[3].strip('\'')] = tokens[2].strip('\'')
				elif tokens[0] == 'state':
					self.all_state.add(tokens[1].strip('\''))
				elif tokens[0] == 'river':
					self.all_river.add(tokens[1].strip('\''))
				elif tokens[0] == 'mountain':
					self.all_mount.add(tokens[1].strip('\''))
		return

	def is_city(self, token):
		token = token.strip('\'')
		return token in self.all_city

	def is_state(self, token):
		token = token.strip('\'')
		return token in self.all_state

	def is_country(self, token):
		token = token.strip('\'')
		return token in self.all_country

	def is_place(self, token):
		token = token.strip('\'')
		return token in self.all_mount

	def is_river(self, token):
		token = token.strip('\'')
		return token in self.all_river

	def find_entity(self, tokens, type):
		if type.lower() == 'city':
			type_function = self.is_city
		elif type.lower() == 'state':
			type_function = self.is_state
		elif type.lower() == 'place':
			type_function = self.is_place
		elif type.lower() == 'country':
			type_function = self.is_country
		elif type.lower() == 'river':
			type_function = self.is_river
		else:
			return None
		# find unary
		for token in tokens:
			if type_function(token):
				return token
		# find binary
		for idx in range(len(tokens) - 1):
			token = ' '.join(tokens[idx:idx+1])
			if type_function(token):
				return token
		# find trinary
		for idx in range(len(tokens) - 2):
			token = ' '.join(tokens[idx:idx+2])
			if type_function(token):
				return token
		return None


class Dataset(object):
	def __init__(self, dataid, dataname, config, lang_vocab, logic_vocab, shuffle=False, batch_size=None):
		if batch_size is None:
			batch_size = config.batch
		filename = './data/d%d_%s_out.txt' % (dataid, dataname)
		instances = prepro.read_out_file(filename)

		datasize = len(instances)
		self.datasize = datasize
		if shuffle:
			indices = list(range(datasize))
			random.shuffle(indices)
			instances = [instances[i] for i in indices]

		data = []
		labels = []
		# preprocess: convert tokens to id
		for ques, logic in instances:
			ques_ids = map_to_ids(ques, lang_vocab)
			logic_ids = map_to_ids(logic, logic_vocab) + [EOS_token]
			data.append((ques_ids, logic_ids))

		# chunk into batches
		batched_data = [data[i:i + batch_size] for i in range(0, datasize, batch_size)]
		self.batched_label = [labels[i:i + batch_size] for i in range(0, datasize, batch_size)]
		self.batched_data = []
		for batch in batched_data:
			batch_size = len(batch)
			batch = list(zip(*batch))
			assert len(batch) == 2
			# sort by descending order of lens
			# lens = [len(x) for x in batch[0]]
			# batch, orig_idx = sort_all(batch, lens)

			ques, ques_lens = self.get_padded_tensor(batch[0], batch_size)
			logic, logic_lens = self.get_padded_tensor(batch[1], batch_size)

			self.batched_data.append((ques, logic, ques_lens, logic_lens))

	def get_padded_tensor(self, tokens_list, batch_size, max_len=MAXLEN):
		""" Convert tokens list to a padded Tensor. """
		# token_len = max(len(x) for x in tokens_list)
		token_lens = [len(x) for x in tokens_list]
		tokens = np.zeros([batch_size, max_len], dtype=np.int32)
		for i, s in enumerate(tokens_list):
			tokens[i, :len(s)] = np.array(s, dtype=np.int32)
		return tokens, token_lens

def map_to_ids(tokens, vocab):
	ids = [vocab[t] for t in tokens]
	return ids

def sort_all(batch, lens):
	""" Sort all fields by descending order of lens, and return the original indices. """
	unsorted_all = [lens] + [range(len(lens))] + list(batch)
	sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
	return sorted_all[2:], sorted_all[1]

def map_to_tokens(idxs, vocab):
	tokens = []
	for idx in idxs:
		if idx > 2:
			tokens.append(vocab[idx])
		else:
			break
	return tokens

def output_to_file(decode, vocab, filename):
	with open(filename, 'w') as f:
		for logic in decode:
			tokens = map_to_tokens(logic, vocab)
			f.write(str(tokens) + '\n')
