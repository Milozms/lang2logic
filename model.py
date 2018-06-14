import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import utils
from torch import optim

class Encoder(nn.Module):
	def __init__(self, args, word_emb):
		super(Encoder, self).__init__()
		hidden, vocab_size, emb_dim, dropout, num_layers,  maxlen = \
			args['hidden'], args['vocab_size'], args['emb_dim'], args['dropout'], args['num_layers'], args['maxlen']
		self.hidden = hidden
		self.dropout = dropout
		self.maxlen = maxlen

		self.word_emb = word_emb

		self.dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True, dropout=dropout)


	def forward(self, inputs):
		emb_words = self.word_emb(inputs)
		output, hidden = self.gru(emb_words)

		return output, hidden


class Decoder(nn.Module):
	def __init__(self, args, word_emb):
		super(Decoder, self).__init__()
		hidden, vocab_size, emb_dim, dropout, num_layers, maxlen, out_vocab_size = \
			args['hidden'], args['vocab_size'], args['emb_dim'], args['dropout'], args['num_layers'], args['maxlen'], args['out_vocab_size']

		self.word_emb = word_emb
		self.hidden = hidden

		self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
		self.hlinear = nn.Linear(hidden, hidden)
		self.clinear = nn.Linear(hidden, hidden)
		self.olinear = nn.Linear(hidden, out_vocab_size)
		self.hlinear.weight.data.normal_(std=0.001)
		self.clinear.weight.data.normal_(std=0.001)
		self.olinear.weight.data.normal_(std=0.001)

	def forward(self, input, encoder_output, init_state, encoder_mask):
		# input is for one RNN cell
		emb_words = self.word_emb(input)
		output, hidden = self.gru(emb_words, init_state)
		# output: [batch, 1, hidden]
		# encoder_output: [batch, input_len, hidden]
		output_ = output.transpose(1, 2)  # output: [batch, hidden, 1]
		attn = torch.bmm(encoder_output, output_).squeeze()  # [batch, input_len]
		# set the score of padding part to -INF (after soft-max: 0)
		attn.masked_fill_(encoder_mask, -float('inf'))
		attn_weight = F.softmax(attn, dim=1)
		attn_weight = torch.unsqueeze(attn_weight, 1)  # [batch, 1, input_len]
		c = torch.bmm(attn_weight, encoder_output).squeeze()  # [batch, hidden]

		w_h_ = self.hlinear(output)
		w_c_ = self.clinear(c)
		h_att = F.tanh(w_h_ + w_c_)

		out_prob = F.softmax(self.olinear(h_att))
		return out_prob, output, hidden


def get_word_emb(vocab_size, emb_dim, emb_mat = None):
	if emb_mat is not None:
		assert vocab_size, emb_dim == emb_mat.shape
		emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID,
									 _weight=torch.from_numpy(emb_mat).float())
	else:
		emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
		emb.weight.data[1:, :].uniform_(-1.0, 1.0)

	return emb


class Model(object):
	def __init__(self, args, device, emb_mat = None):
		self.out_vocab_size = args['out_vocab_size']
		word_emb = get_word_emb(args['vocab_size'], args['emb_dim'], emb_mat)
		self.encoder = Encoder(args, word_emb)
		self.decoder = Decoder(args, word_emb)
		self.encoder_optimizer = optim.SGD(self.encoder.parameters(), args['lr'])
		self.decoder_optimizer = optim.SGD(self.decoder.parameters(), args['lr'])
		self.device = device
		self.criterion = nn.CrossEntropyLoss()


	def train(self, inputs, targets):
		self.encoder.train()
		self.decoder.train()
		self.encoder_optimizer.zero_grad()
		self.decoder_optimizer.zero_grad()

		batch, input_length = inputs.size()
		batch_, target_length = targets.size()
		assert batch == batch_

		inputs.to(self.device)
		targets.to(self.device)

		encoder_mask = torch.eq(inputs, utils.PAD_ID)  # padding part: 1  [batch, input_len]

		encoder_outputs, encoder_hidden = self.encoder(inputs)

		decoder_input = torch.tensor([[utils.SOS_token] for cnt in batch], device=self.device)  # [batch, 1]
		decoder_hidden = encoder_hidden

		outputs = torch.zeros(batch, target_length, self.out_vocab_size)

		# Teacher forcing: Feed the target as the next input
		for di in range(target_length):
			out_prob, decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_outputs, decoder_hidden, encoder_mask)
			# out_prob: [batch, out_vocab_size]
			# decoder_output: [batch, 1, hidden]
			# decoder_hidden: [nlayer, batch, hidden]
			outputs[:, di, :] = out_prob
			decoder_input = targets[di]  # Teacher forcing

		loss, ts_cnt = self.comptute_loss(targets, outputs)

		loss.backward()
		self.encoder_optimizer.step()
		self.decoder_optimizer.step()

		return loss.item() / ts_cnt


	def decode(self, inputs, targets, maxlen=utils.MAXLEN):
		self.encoder.eval()
		self.decoder.eval()

		batch, input_length = inputs.size()
		with torch.no_grad():
			inputs.to(self.device)
			targets.to(self.device)
			encoder_mask = torch.eq(inputs, utils.PAD_ID)  # padding part: 1  [batch, input_len]

			encoder_outputs, encoder_hidden = self.encoder(inputs)

			decoder_input = torch.tensor([[utils.SOS_token] for cnt in batch], device=self.device)  # [batch, 1]
			decoder_hidden = encoder_hidden

			outputs = torch.zeros(batch, maxlen, self.out_vocab_size)

			decoded_words = []

			for di in range(maxlen):
				out_prob, decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_outputs, decoder_hidden, encoder_mask)
				# out_prob: [batch, out_vocab_size]
				# decoder_output: [batch, 1, hidden]
				# decoder_hidden: [nlayer, batch, hidden]
				topv, topi = out_prob.topk(1, dim=1)
				decoded_words.append(topi.squeeze().tolist())
				decoder_input = topi.squeeze().detach()
				outputs[:, di, :] = out_prob.squeeze(1)

			loss, ts_cnt = self.comptute_loss(targets, outputs)

			return decoded_words, loss.item() / ts_cnt


	def comptute_loss(self, targets, outputs):
		target_mask = torch.eq(targets, utils.PAD_ID).eq(utils.PAD_ID)  # padding part: 0  [batch, maxlen]
		outputs_unfold = outputs.view(-1, self.decoder.hidden) 	# [batch*target_length, hidden]
		targets_unfold = targets.view(-1)					# [batch*target_length]
		loss_unfold = self.criterion(outputs_unfold, targets_unfold, reduce=False) 	# [batch*target_length]
		mask_unfold = target_mask.view(-1).float()
		ts_cnt = mask_unfold.sum()
		valid_losses = torch.mul(loss_unfold, mask_unfold)
		loss = valid_losses.sum()
		return loss, ts_cnt