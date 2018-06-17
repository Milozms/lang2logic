import re
import json
import linecache
import pickle
import numpy as np
from tqdm import tqdm
import utils
import d1base
import d2base

def tokenize(logic):
	# pattern = re.compile('(?:\))|(?:,)|(?:[^\(\)\+\\\,]*\()|(?:[^\(\)\+\\\,]+)')
	pattern = re.compile('(?:\'[^\'\(\)\,\+\\\]+\')|[^\s\'\(\)\,\+\\\]+|(?:[\+\\\]+ *)')
	tokens_ = pattern.findall(logic)
	# if '\'' in logic:
	# 	print(tokens_)
	tokens = []
	for idx, token in enumerate(tokens_):
		if len(token) == 1 and token.isupper():
			tokens.append(token)
			pass
		elif idx > 0 and tokens_[idx-1][-2:] == 'id':
			tokens.append('<%s>' % tokens[idx-1][:-2].upper())
		elif idx > 1 and tokens_[idx-2] == 'cityid':
			tokens.append('<STATE_ABBR>')
		else:
			tokens.append(token)
	return tokens[1:]

def read_out_file(filename, dataid=1):
	if dataid == 2:
		return d2base.read_out_file(filename)
	f = open(filename, 'r')
	dataset = []
	for line in f.readlines():
		line = line.strip()
		q_end_idx = line.find(', answer')
		# tokens = line.strip().split(' ')
		# question = tokens[0][6:-1]  # strip 'parse(' and ','
		# logic = tokens[1][:-2]  # strip ').'
		question = line[6:q_end_idx]
		logic = line[q_end_idx+2:-2]
		ques_tokens = question[1:-1].split(',')
		logic_tokens = tokenize(logic)
		dataset.append((ques_tokens, logic_tokens))
	f.close()
	return dataset

def read_in_file(filename):
	f = open(filename, 'r')
	dataset = []
	for line in f.readlines():
		question = line.strip()
		ques_tokens = question[1:-1].split(',')
		dataset.append(ques_tokens)
	f.close()
	return dataset

def build_lang_vocab(data_id=1):
	maxlen = 0
	vocab = set()
	for data in ['train', 'valid', 'test']:
		filename = './data/d%d_%s_in.txt' % (data_id, data)
		questions = read_in_file(filename)
		for ques in questions:
			for tok in ques:
				vocab.add(tok)
			if len(ques) > maxlen:
				maxlen = len(ques)
	vocabfile = './vocab/d%d_vocab_lang.json' % data_id
	with open(vocabfile, 'w') as f:
		json.dump(list(vocab), f)
	print('Size of language vocab: %d' % len(vocab))
	print('Max length of questions: %d' % maxlen)
	return vocab

def build_logic_vocab(data_id=1):
	vocab = set()
	maxlen = 0
	for data in ['train', 'valid']:
		filename = './data/d%d_%s_out.txt' % (data_id, data)
		dset = read_out_file(filename)
		for ques, logic in dset:
			for tok in logic:
				vocab.add(tok)
			if len(logic) > maxlen:
				maxlen = len(logic)
	vocabfile = './vocab/d%d_vocab_logic.json' % data_id
	num_of_args = {}
	for pred in vocab:
		num_of_args[pred] = 0
	with open(vocabfile, 'w') as f:
		json.dump(list(vocab), f)
	print('Size of logic vocab: %d' % len(vocab))
	print('Max length of logic: %d' % maxlen)
	return vocab

def build_emb(dataid):
	vocabfile = './vocab/d%d_vocab_lang.json' % dataid
	emb_file = '/Users/zms/Documents/学习资料/NLP/glove.840B.300d.txt'
	with open(vocabfile, 'r') as f:
		vocab = json.load(f)
	vocab = utils.vocab_prefix + vocab
	vocab_size = len(vocab)
	emb_dim = 300
	word2id = {}
	emb = np.random.randn(vocab_size, emb_dim)
	for idx, word in enumerate(vocab):
		word2id[word] = idx
	for line in tqdm(linecache.getlines(emb_file)):
		tokens = line.strip().split()
		word = tokens[0]
		vec = [float(x) for x in tokens[-emb_dim:]]
		if word in word2id:
			idx = word2id[word]
			emb[idx, :] = np.array(vec, dtype=np.float32)
	with open('./vocab/d%d_word2id.json' % dataid, 'w') as f:
		json.dump(word2id, f)
	with open('./vocab/d%d_emb.pkl' % dataid, 'wb') as f:
		pickle.dump(emb, f)

	return word2id, emb

if __name__ == '__main__':
	# dataset = read_out_file('./data/d1_valid_out.txt')
	# build_lang_vocab(2)
	build_logic_vocab(2)
	# build_emb(2)
	# d2base = d2base.D2base()
	# d1base.read_and_restore('./data/d1_valid_out.txt')