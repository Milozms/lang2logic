import numpy as np
import prepro
import random
import re
import json

type2entities = {'language':set(['c++'])}

def tokenize(logic):
	# pattern = re.compile('(?:\))|(?:,)|(?:[^\(\)\+\\\,]*\()|(?:[^\(\)\+\\\,]+)')
	pattern = re.compile('(?:\'[^\'\(\)\,\\\]+\')|[^\s\'\(\)\,\\\]+|(?:[\+\\\]+ *)')
	tokens_ = pattern.findall(logic)
	# if '\'' in logic:
	# 	print(tokens_)
	tokens = []
	for idx, token in enumerate(tokens_):
		if len(token) == 1 and token.isupper():
			tokens.append(token)
		elif len(token) > 1 and token[0] == '_':
			# tokens.append('V')
			tokens.append(token)
		elif idx > 0 and tokens_[idx-1][-2:] == 'id':
			tokens.append('<%s>' % tokens[idx-1][:-2].upper())
		elif idx > 1 and tokens_[idx-2] == 'const':
			if True:
				m = re.search('(\w+)\(\w+,\w+\),const\(\w+,%s\)' % re.escape(token), logic)
				if m is None:
					tokens.append('<CONST>')
				else:
					const_type = m.group(1)
					tokens.append('<%s>' % const_type.upper())
					if const_type not in type2entities:
						type2entities[const_type] = set()
					type2entities[const_type].add(token.strip('\''))
			# except:
			# 	tokens.append('<CONST>')
		elif token.isnumeric():
			tokens.append('<NUMBER>')
		else:
			tokens.append(token)
	return tokens

def read_out_file(filename):
	f = open(filename, 'r')
	dataset = []
	for line in f.readlines():
		line = line.strip()
		m = re.match(r'^.*parse\((\[.*\]), *answer(\(.*\))\).*$', line)
		question = m.group(1)
		logic = m.group(2)
		ques_tokens = question.strip('[]').split(',')
		logic_tokens = tokenize(logic)
		dataset.append((ques_tokens, logic_tokens))
	f.close()
	return dataset

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

class D2base(object):
	def __init__(self):
		self.read_d2_base()
		vocabfile = './vocab/d2_vocab_logic.json'
		with open(vocabfile, 'r') as f:
			logic_vocab = json.load(f)
		self.num_of_args = {}
		for pred in logic_vocab:
			self.num_of_args[pred] = 0
		# self.load_num_of_args()
		self.find_num_of_args('./data/d2_train_out.txt')

	def read_d2_base(self):
		self.type2entities = {}
		filename = './base/d2_base'
		with open(filename, 'r', encoding='ascii', errors='ignore') as f:
			for idx, line in enumerate(f.readlines()):
				if idx < 30:
					continue
				line = line.strip()
				m = re.match('(\w+)\(\'[^\']+\', *\'([\w\+]+)\'\)', line)
				if m is not None:
					entity_type = m.group(1)
					entity = m.group(2)
					if entity_type not in self.type2entities:
						self.type2entities[entity_type] = set()
					self.type2entities[entity_type].add(entity)
		return


	def find_entity(self, tokens, type):
		type = type.lower()
		if type == 'number':
			for token in tokens:
				if token.isnumeric():
					return token
		if type not in type2entities:
			return None
		all_entities = type2entities[type]
		# find trinary
		for idx in range(len(tokens) - 2):
			token = ' '.join(tokens[idx:idx+3])
			if token in all_entities:
				return '\'%s\'' % token
		# find binary
		for idx in range(len(tokens) - 1):
			token = ' '.join(tokens[idx:idx + 2])
			if token in all_entities:
				return '\'%s\'' % token
		# find unary
		for token in tokens:
			if token in all_entities:
				return '\'%s\'' % token
			else:
				for ent in all_entities:
					if token.lower() == ent.lower():
						return '\'%s\'' % ent
		return None

	def find_arguments(self, logic, pred):
		pred_idx = logic.find(pred)
		if pred_idx < 0:
			return None
		logic = logic[pred_idx:]
		start_idx = logic.find('(')
		if start_idx < 0:
			return None
		if len(pred) != start_idx:
			return
		args = []
		brk_stack = []
		cur_tok = ''
		for idx, c in enumerate(logic):
			if c == '(':
				cur_tok = ''
				brk_stack.append((idx, c))
			elif c == ')':
				if len(brk_stack) == 0:
					return len(args)
				lidx, lb = brk_stack.pop()
				if len(brk_stack) == 0:
					if len(cur_tok) > 0:
						args.append(cur_tok)
						cur_tok = ''
				elif len(brk_stack) == 1:
					content = logic[lidx + 1:idx]
					args.append(content)
			elif c == ',':
				if len(brk_stack) == 0:
					return len(args)
				if len(cur_tok) > 0 and len(brk_stack) == 1:
					args.append(cur_tok)
					cur_tok = ''
			else:
				if len(brk_stack) == 1:
					cur_tok += c
		return len(args)

	def find_num_of_args(self, filename):
		f = open(filename, 'r')
		all_logics = []
		for line in f.readlines():
			line = line.strip()
			q_end_idx = line.find(', answer')
			question = line[6:q_end_idx]
			logic = line[q_end_idx + 9:-3]
			all_logics.append(logic)
		for pred in self.num_of_args.keys():
			if pred in ['req_deg', 'req_exp']:
				a = 0
			if len(pred) > 3 and pred[-3:] == 'est':
				self.num_of_args[pred] = -1
				continue
			for logic in all_logics:
				val = self.find_arguments(logic, pred)
				if val is not None:
					self.num_of_args[pred] = val
					break
		f.close()
		self.num_of_args['capital'] = 1
		with open('./vocab/d2_num_of_args.json', 'w') as f:
			json.dump(self.num_of_args, f)

	def load_num_of_args(self):
		with open('./vocab/num_of_args.json', 'r') as f:
			self.num_of_args = json.load(f)

	def restore_logic(self, tokens, question):
		stack = []
		cur_city = ''
		for token in tokens:
			nargs = self.num_of_args[token]
			if token[0] == '<':
				token_type = token[1:-1].lower()
				entity = self.find_entity(question, token_type)
				if entity is not None:
					if token_type == 'state_abbr':
						if len(cur_city) > 0:
							entity = self.city2state[cur_city]
						else:
							entity = self.state_abbr[entity.strip('\'')]
					elif token_type == 'place':
						entity = '\'mount %s\'' % entity
					elif token_type == 'country' and entity in ['us', '\'united states\'']:
						entity = 'usa'
					elif token_type == 'city':
						cur_city = entity.strip('\'')
					token = entity
				else:
					if token_type == 'state_abbr':
						token = '_'
			if nargs == 0 and len(stack) > 0 and stack[-1][1] == 1:
				pred, _ = stack.pop()
				new_token = '%s(%s)' % (pred, token)
				stack.append((new_token, 0))
			elif nargs == 0 and len(stack) > 1 and stack[-1][1] == 0 and stack[-2][1] == 2:
				arg1, _ = stack.pop()
				pred, _ = stack.pop()
				new_token = '%s(%s,%s)' % (pred, arg1, token)
				stack.append((new_token, 0))
			elif nargs == 0 and len(stack) > 2 and stack[-1][1] == 0 and stack[-2][1] == 0 and stack[-3][1] == 3:
				arg2, _ = stack.pop()
				arg1, _ = stack.pop()
				pred, _ = stack.pop()
				new_token = '%s(%s,%s,%s)' % (pred, arg1, arg2, token)
				stack.append((new_token, 0))
			else:
				stack.append((token, nargs))
			while len(stack) > 1:
				if len(stack) > 1 and stack[-1][1] == 0 and stack[-2][1] == 1:
					arg , _ = stack.pop()
					pred, _ = stack.pop()
					new_token = '%s(%s)' % (pred, arg)
					stack.append((new_token, 0))
				elif len(stack) > 2 and stack[-1][1] == 0 and stack[-2][1] == 0 and stack[-3][1] == 2:
					arg2, _ = stack.pop()
					arg1, _ = stack.pop()
					pred, _ = stack.pop()
					new_token = '%s(%s,%s)' % (pred, arg1, arg2)
					stack.append((new_token, 0))
				else:
					break
		tokens = [x[0] for x in stack]
		if len(stack) > 2:
			arg = '(%s)' % ','.join(tokens[1:])
		else:
			arg = tokens[1]
		result = 'answer(%s,%s)' % (tokens[0], arg)
		return result

	def read_and_restore(self, filename):
		f = open(filename, 'r')
		outf = open('./data/result2.txt', 'w')
		for idx, line in enumerate(f.readlines()):
			line = line.strip()
			m = re.match(r'^.*parse\((\[.*\]), *(answer\(.*\))\).*$', line)
			question = m.group(1)
			logic = m.group(2)
			ques_tokens = question.strip('[]').split(',')
			logic_tokens = tokenize(logic)[1:]
			result = self.restore_logic(logic_tokens, ques_tokens)
			if logic != result:
				outf.write('%d: %s\n%s\n%s\n\n' % (idx, question, logic, result))
		f.close()
		outf.close()

if __name__ == '__main__':
	# build_logic_vocab(2)
	# prepro.build_emb(2)
	read_out_file('./data/d2_train_out.txt')
	d2base = D2base()
	d2base.read_and_restore('./data/d2_valid_out.txt')
