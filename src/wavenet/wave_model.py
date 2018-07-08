import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys


class WavenetModel(nn.Module):

	def __init__(self,
				 layers=10,
				 dilation_channels=32,
				 stacks=3,
				 residual_channels=32,
				 skip_channels=256,
				 end_channels=256,
				 classes=256,
				 output_length=256,
				 kernel_size=2,
				 cin_channels=-1,
				 dtype=torch.FloatTensor,
				 bias=False):
		"""

        Args:
            layers ():
            dilation_channels ():
            stacks ():
            residual_channels ():
            skip_channels ():
            end_channels ():
            classes ():
            output_length ():
            kernel_size ():
            cin_channels ():
            dtype ():
            bias ():
        """
		super(WavenetModel, self).__init__()

		self.layers = layers
		self.stacks = stacks
		self.dilation_channels = dilation_channels
		self.residual_channels = residual_channels
		self.skip_channels = skip_channels
		self.classes = classes
		self.kernel_size = kernel_size
		self.dtype = dtype

		self.filters = nn.ModuleList()
		self.gates = nn.ModuleList()
		self.residuals = nn.ModuleList()
		self.skips = nn.ModuleList()

		# for local conditioning
		self.cin_channels = cin_channels
		if cin_channels != -1:
			self.lc_filters = nn.ModuleList()
			self.lc_gates = nn.ModuleList()

		receptive_field = 0

		self.causal = nn.Conv1d(in_channels=1,
								out_channels=residual_channels,
								kernel_size=1,
								bias=1)

		# TODO: add upsampling layers
		# self.upsamples = nn.ModuleList()
		# for scales:
		# nn.ConvTranspose2d(in_channels = 1, out_channels = 1, kernel_size=(freq x scale), stride=(1 x scale), bias=1, padding=1)

		for s in range(stacks):
			dilation = 1
			additional_scope = kernel_size - 1
			for i in range(layers):

				# filter convolution
				self.filters.append(nn.Conv1d(in_channels=residual_channels,
											  out_channels=dilation_channels,
											  kernel_size=kernel_size,
											  dilation=dilation,
											  bias=bias))

				# gate convolution
				self.gates.append(nn.Conv1d(in_channels=residual_channels,
											out_channels=dilation_channels,
											kernel_size=kernel_size,
											dilation=dilation,
											bias=bias))

				# Local conditioning convolutions for mel spec
				if cin_channels != -1:
					self.lc_filters.append(nn.Conv1d(in_channels=cin_channels,
													 out_channels=dilation_channels,
													 kernel_size=kernel_size,
													 bias=bias))

					self.lc_gates.append(nn.Conv1d(in_channels=cin_channels,
												   out_channels=dilation_channels,
												   kernel_size=kernel_size,
												   bias=bias))

				# residual convolutions
				self.residuals.append(nn.Conv1d(in_channels=dilation_channels,
												out_channels=residual_channels,
												kernel_size=1,
												dilation=dilation,
												bias=bias))

				# skip connection convolutions
				self.skips.append(nn.Conv1d(in_channels=dilation_channels,
											out_channels=skip_channels,
											kernel_size=1,
											bias=bias))

				receptive_field += additional_scope
				dilation *= 2
				additional_scope *= 2

		self.end1 = nn.Conv1d(in_channels=skip_channels,
							  out_channels=end_channels,
							  kernel_size=1,
							  bias=True)

		self.end2 = nn.Conv1d(in_channels=end_channels,
							  out_channels=classes,
							  kernel_size=1,
							  bias=True)

		self.output_length = output_length
		self.receptive_field = receptive_field

	def forward(self, y, c=None):
		"""

        Args:
            y (): audio data
            c (): local conditioned data ie. mel spectrogram

        Returns:
        	audio data

        """

		x = self.causal(y)
		skip = 0

		# TODO: local conditioning upsample layer (currently expanding the input)
		for i in range(self.layers * self.stacks):

			# residual is r
			r = x
			f = self.filters[i](x)
			g = self.gates[i](x)

			# Local conditioning convolutions
			if self.cin_channels != -1:
				f_c = self.lc_filters[i](c)
				g_c = self.lc_gates[i](c)
				f = f + f_c[:, :, :f.size(-1)]  # find a better solution to adjust time?
				g = g + g_c[:, :, :g.size(-1)]

			f = F.tanh(f)
			g = F.sigmoid(g)
			x = f * g

			# skip connections
			s = self.skips[i](x)
			try:
				skip = skip[:, :, -s.size(2):]
			except:
				skip = 0

			skip = s + skip

			# add residual and dilated output
			x = self.residuals[i](x)
			x = x + r[:, :, -x.size(2):]

		# last layers, x is the input of softmax
		x = F.relu(skip)
		x = self.end1(x)

		x = F.relu(x)
		x = self.end2(x)

		return x

	def generate(self, sample_size, first_samples=None):
		"""
		Start with zeros, produces 1 by 1.
		TODO: add first samples

		Args:
			sample_size (int): Required length of audio
			first_samples ():

		Returns:
			Generated audio with size = sample_size
		"""

		output_length = self.output_length
		receptive_length = self.receptive_field
		input_length = receptive_length - 1 + output_length

		if sample_size <= input_length:
			return sys.exit("Sample size is short!")

		init = np.zeros(input_length)
		init = torch.tensor(init).float().view(1, 1, -1)

		while init.size(-1) != sample_size:
			x = init[:, :, -input_length:]
			gen_out = self.forward(x)
			last = gen_out[:, :, -1]
			decoded = util.one_hot_decode(last)
			init = torch.cat((init.float(), decoded.float().view(1, 1, -1)), 2)

		return init
