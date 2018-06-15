import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torch import optim
from tqdm import tqdm

class Encoder(nn.Module):
	def __init__(self, args, word_emb):
		super(Encoder, self).__init__()
		hidden, vocab_size, emb_dim, dropout, num_layers = \
			args['hidden'], args['vocab_size'], args['emb_dim'], args['dropout'], args['num_layers']
		self.hidden = hidden
		self.dropout = dropout

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
		hidden, vocab_size, emb_dim, dropout, num_layers, out_vocab_size = \
			args['hidden'], args['vocab_size'], args['emb_dim'], args['dropout'], args['num_layers'], args['out_vocab_size']

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

		output = output.squeeze()
		w_h_ = self.hlinear(output)
		w_c_ = self.clinear(c)
		h_att = F.tanh(w_h_ + w_c_)

		out_prob = F.softmax(self.olinear(h_att), dim=1)
		return out_prob, output, hidden


def get_word_emb(vocab_size, emb_dim, emb_mat = None):
	if emb_mat is not None:
		assert vocab_size, emb_dim == emb_mat.shape
		emb = nn.Embedding(vocab_size, emb_dim, padding_idx=utils.PAD_ID,
									 _weight=torch.from_numpy(emb_mat).float())
	else:
		emb = nn.Embedding(vocab_size, emb_dim, padding_idx=utils.PAD_ID)
		emb.weight.data[1:, :].uniform_(-1.0, 1.0)

	return emb


class Model(object):
	def __init__(self, args, device, emb_mat = None):
		self.args = args
		self.out_vocab_size = args['out_vocab_size']
		word_emb = get_word_emb(args['vocab_size'], args['emb_dim'], emb_mat)
		self.max_grad_norm = args['max_grad_norm']
		self.encoder = Encoder(args, word_emb)
		self.decoder = Decoder(args, word_emb)
		# self.encoder_optimizer = optim.RMSprop(self.encoder.parameters(), args['lr'], alpha=0.95)
		# self.decoder_optimizer = optim.RMSprop(self.decoder.parameters(), args['lr'], alpha=0.95)
		self.encoder_optimizer = optim.Adam(self.encoder.parameters(), args['lr'])
		self.decoder_optimizer = optim.Adam(self.decoder.parameters(), args['lr'])
		self.device = device
		self.criterion = nn.CrossEntropyLoss(reduce=False)


	def train_instance(self, input, target):
		self.encoder.train()
		self.decoder.train()
		self.encoder_optimizer.zero_grad()
		self.decoder_optimizer.zero_grad()

		input_length = input.size(0)
		target_length = target.size(0)

		encoder_outputs = torch.zeros(target_length, self.out_vocab_size, device=self.device)

		loss = 0

		for ei in range(input_length):
			encoder_output, encoder_hidden = self.encoder(
				input[ei], encoder_hidden)
			encoder_outputs[ei] = encoder_output[0, 0]

		decoder_input = torch.tensor([[utils.SOS_token]], device=device)

		decoder_hidden = encoder_hidden


		for di in range(target_length):
			out_prob, decoder_output, decoder_hidden = self.decoder(
					decoder_input, decoder_hidden, encoder_outputs)
			loss += self.criterion(decoder_output, target[di])
			decoder_input = target[di]  # Teacher forcing


		loss.backward()

		self.encoder_optimizer.step()
		self.decoder_optimizer.step()

		return loss.item() / target_length

	def train_batch(self, inputs, targets):
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

		decoder_input = torch.tensor([[utils.SOS_token] for cnt in range(batch)], device=self.device)  # [batch, 1]
		decoder_hidden = encoder_hidden

		outputs = torch.zeros(batch, target_length, self.out_vocab_size)

		# Teacher forcing: Feed the target as the next input
		for di in range(target_length):
			out_prob, decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_outputs, decoder_hidden, encoder_mask)
			# out_prob: [batch, out_vocab_size]
			# decoder_output: [batch, hidden]
			# decoder_hidden: [nlayer, batch, hidden]
			outputs[:, di, :] = out_prob
			decoder_input = targets[:, di].unsqueeze(1)  # Teacher forcing

		loss, ts_cnt = self.comptute_loss(targets, outputs)

		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
		torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)
		self.encoder_optimizer.step()
		self.decoder_optimizer.step()

		return loss.item() / ts_cnt


	def decode_batch(self, inputs, targets, maxlen=utils.MAXLEN):
		self.encoder.eval()
		self.decoder.eval()

		batch, input_length = inputs.size()
		with torch.no_grad():
			inputs.to(self.device)
			targets.to(self.device)
			encoder_mask = torch.eq(inputs, utils.PAD_ID)  # padding part: 1  [batch, input_len]

			encoder_outputs, encoder_hidden = self.encoder(inputs)

			decoder_input = torch.tensor([[utils.SOS_token] for cnt in range(batch)], device=self.device)  # [batch, 1]
			decoder_hidden = encoder_hidden

			outputs = torch.zeros(batch, maxlen, self.out_vocab_size)

			decoded_words = []

			for di in range(maxlen):
				out_prob, decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_outputs, decoder_hidden, encoder_mask)
				# out_prob: [batch, out_vocab_size]
				# decoder_output: [batch, hidden]
				# decoder_hidden: [nlayer, batch, hidden]
				topv, topi = out_prob.topk(1, dim=1)
				decoded_words.append(topi.squeeze().tolist())
				decoder_input = topi.detach()
				outputs[:, di, :] = out_prob.squeeze(1)

			loss, ts_cnt = self.comptute_loss(targets, outputs)

			return decoded_words, loss.item() / ts_cnt

	def eval(self, dataset):
		loss = 0
		decoded = []
		for idx, batch in enumerate(tqdm(dataset.batched_data)):
			ques, logic = batch
			decoded_batch, loss_batch = self.decode_batch(ques, logic)
			loss += loss_batch
			decoded += decoded_batch
		loss /= len(dataset.batched_data)
		return loss, decoded


	def comptute_loss(self, targets, outputs):
		target_len = targets.size(1)
		outputs = outputs[:, :target_len, :]
		target_mask = torch.eq(targets, utils.PAD_ID).eq(utils.PAD_ID)  # padding part: 0  [batch, maxlen]
		outputs_unfold = outputs.contiguous().view(-1, self.out_vocab_size) 	# [batch*target_length, out_vocab_size]
		targets_unfold = targets.view(-1)					# [batch*target_length]
		loss_unfold = self.criterion(outputs_unfold, targets_unfold) 	# [batch*target_length]
		mask_unfold = target_mask.view(-1).float()
		ts_cnt = mask_unfold.sum()
		valid_losses = torch.mul(loss_unfold, mask_unfold)
		loss = valid_losses.sum()
		return loss, ts_cnt

	def save(self, filename, epoch):
		params = {
			'encoder': self.encoder.state_dict(),
			'decoder': self.decoder.state_dict(),
			'config': self.args,
			'epoch': epoch
		}
		try:
			torch.save(params, filename)
			print("model saved to {}".format(filename))
		except BaseException:
			print("[Warning: Saving failed... continuing anyway.]")
