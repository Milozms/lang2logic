import re
import json
import linecache
import pickle
import numpy as np
from tqdm import tqdm
import utils

def tokenize(logic):
	pattern = re.compile('(?:\))|(?:,)|(?:[^\(\)\+\\\,]*\()|(?:[^\(\)\+\\\,]+)')
	tokens = pattern.findall(logic)
	return tokens

def read_out_file(filename):
	f = open(filename, 'r')
	dataset = []
	for line in f.readlines():
		tokens = line.strip().split(' ')
		question = tokens[0][6:-1]  # strip 'parse(' and ','
		logic = tokens[1][:-2]  # strip ').'
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
	vocabfile = './vocab/vocab_lang.json'
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
	vocabfile = './vocab/vocab_logic.json'
	with open(vocabfile, 'w') as f:
		json.dump(list(vocab), f)
	print('Size of logic vocab: %d' % len(vocab))
	print('Max length of logic: %d' % maxlen)
	return vocab

def build_emb():
	vocabfile = './vocab/vocab_lang.json'
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
	with open('./vocab/word2id.json', 'w') as f:
		json.dump(word2id, f)
	with open('./vocab/emb.pkl', 'wb') as f:
		pickle.dump(emb, f)

	return word2id, emb

if __name__ == '__main__':
	# dataset = readfile('./data/d1_valid_out.txt')
	# build_lang_vocab(1)
	# build_logic_vocab(1)
	build_emb()