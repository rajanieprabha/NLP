import torch
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from src.dataset import WavenetLoader
import src.wavenet.wave_util as util


def train(dataloader, model, optimizer, global_step, hparams, condition=False):
	model.train()

	for d in dataloader:
		# TODO: train by conditioning on mel
		# if condition:
		# x, mel, target
		for x, target in d:
			x = x.float()
			x = Variable(x)
			x, target = Variable(x), Variable(target.long())
			# if model.is_cuda:
			# x = x.cuda()
			# target = target.cuda()
			y = model(x)
			y = y.view(-1, hparams.classes)  # number of classes

			loss = F.cross_entropy(y, target)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if global_step % 100 == 0:
			print("Step: {} Loss: {}".format(global_step, loss))

		global_step += 1

	return global_step


def val(dataloader, model, global_step, hparams):
	model.val()

	all_loss = []
	with torch.no_grad():
		for d in dataloader:
			for x, target in d:
				x = x.float()
				x = Variable(x)
				x, target = Variable(x), Variable(target.long())
				# if model.is_cuda:
				# x = x.cuda()
				# target = target.cuda()
				y = model(x)
				y = y.view(-1, model.classes)  # number of classes
				loss = F.cross_entropy(y.squeeze(), target.squeeze())
				all_loss.append(loss)

	avg_loss = torch.mean(all_loss)

	print("Step: {} Loss: {}".format(global_step, avg_loss))

	return avg_loss


def train_and_evaluate(dataset, hparams):
	model = util.fetch_model(hparams)
	dataloader = util.fetch_dataloader(dataset, model, hparams)
	optimizer = util.fetch_optimizer(model, hparams)

	global_step = 0
	best_metric = 0.0

	for epoch in range(hparams.num_epoch):

		print("Epoch: {}".format(epoch))

		global_step = train(dataloader['train'], model, optimizer, global_step, hparams)

		metric = val(dataloader['val'], model, global_step, hparams)

		is_best = False
		if metric >= best_metric:
			is_best = True
			best_metric = metric

		if is_best:
			print("yey")
