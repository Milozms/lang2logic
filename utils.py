import torch
import prepro
import random

PAD_ID = 0
MAXLEN = 100 	# 23, 85
SOS_token = 1
EOS_token = 2
vocab_prefix = ['<PAD>', '<SOS>', '<EOS>']

class Dataset(object):
	def __init__(self, dataid, dataname, args, lang_vocab, logic_vocab, device, shuffle=False, batch_size=None):
		if batch_size is None:
			batch_size = args['batch_size']
		self.device = device
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
			ques_ids = map_to_ids(ques, lang_vocab) + [EOS_token]
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

			ques = self.get_padded_tensor(batch[0], batch_size)
			logic = self.get_padded_tensor(batch[1], batch_size)

			self.batched_data.append((ques, logic))

	def get_padded_tensor(self, tokens_list, batch_size):
		""" Convert tokens list to a padded Tensor. """
		token_len = max(len(x) for x in tokens_list)
		tokens = torch.zeros(batch_size, token_len, dtype=torch.long).fill_(PAD_ID)
		for i, s in enumerate(tokens_list):
			tokens[i, :len(s)] = torch.tensor(s, dtype=torch.long)
		return tokens

def map_to_ids(tokens, vocab):
	ids = [vocab[t] if t in vocab else PAD_ID for t in tokens]
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