
class D1base(object):
	def __init__(self):
		self.read_d1_base()
		vocabfile = './vocab/vocab_logic.json'
		with open(vocabfile, 'r') as f:
			logic_vocab = json.load(f)
		self.num_of_args = {}
		for pred in logic_vocab:
			self.num_of_args[pred] = 0
		# self.load_num_of_args()

	def read_d1_base(self):
		self.all_city = set()
		self.city2state = {}
		self.all_state = set()
		self.state_abbr = {}
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
					self.state_abbr[tokens[1].strip('\'')] = tokens[2].strip('\'')
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
		elif type.lower() == 'state_abbr':
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
				return '\'%s\'' % token
		# find trinary
		for idx in range(len(tokens) - 2):
			token = ' '.join(tokens[idx:idx+2])
			if type_function(token):
				return '\'%s\'' % token
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
			if pred == 'city':
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
		with open('./vocab/num_of_args.json', 'w') as f:
			json.dump(self.num_of_args, f)

	def load_num_of_args(self):
		with open('./vocab/num_of_args.json', 'r') as f:
			self.num_of_args = json.load(f)

	def restore_logic(self, tokens, question):
		stack = []
		for token in tokens:
			nargs = self.num_of_args[token]
			if token[0] == '<':
				token_type = token[1:-1].lower()
				entity = self.find_entity(question, token_type)
				if entity is not None:
					if token_type == 'state_abbr':
						entity = self.state_abbr[entity.strip('\'')]
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
		for idx in range(len(stack) - 1, -1, -1):
			token = stack[idx][0]
			if token == 'count':
				args = [x[0] for x in stack[idx+1:]]
				if len(args) > 3:
					arg2 = '(%s)' % ','.join(args[1:-1])
				else:
					arg2 = args[1]
				new_token = '%s(%s,%s,%s)' % (token, args[0], arg2, args[-1])
				stack = stack[:idx] + [(new_token, 0)]
			elif token in ['most', 'fewest', 'sum']:
				args = [x[0] for x in stack[idx + 1:]]
				if len(args) > 3:
					arg3 = '(%s)' % ','.join(args[2:])
				else:
					arg3 = args[-1]
				new_token = '%s(%s,%s,%s)' % (token, args[0], args[1], arg3)
				stack = stack[:idx] + [(new_token, 0)]
			elif len(token) > 3 and token[-3:] == 'est':
				args = [x[0] for x in stack[idx + 1:]]
				if len(args) > 2:
					arg2 = '(%s)' % ','.join(args[1:])
				else:
					arg2 = args[1]
				new_token = '%s(%s,%s)' % (token, args[0], arg2)
				stack = stack[:idx] + [(new_token, 0)]
		tokens = [x[0] for x in stack]
		if len(stack) > 2:
			arg = '(%s)' % ','.join(tokens[1:])
		else:
			arg = tokens[1]
		result = 'answer(A,%s)' % arg
		return result

	def read_and_restore(self, filename):
		f = open(filename, 'r')
		outf = open('./data/result.txt', 'w')
		for line in f.readlines():
			line = line.strip()
			q_end_idx = line.find(', answer')
			question = line[6:q_end_idx]
			logic = line[q_end_idx + 2:-2]
			ques_tokens = question[1:-1].split(',')
			logic_tokens = prepro.tokenize(logic)
			result = self.restore_logic(logic_tokens, ques_tokens)
			outf.write('%s\n%s\n\n' % (logic, result))
		f.close()
		outf.close()

