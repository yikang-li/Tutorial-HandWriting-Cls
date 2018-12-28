# pytorch related packages
import torch
import torch.nn as nn
import torch.nn.functional as F 


# Model Definition
class Net(nn.Module):
	def __init__(self, num_classes):
		super(Net, self).__init__()
		self.conv = nn.Sequential(
				nn.Conv2d(3, 20, 3, 1),
				nn.BatchNorm2d(20, ),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=3, stride=2),
				nn.Conv2d(20, 40, 3, 1),
				nn.BatchNorm2d(40, ),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=3, stride=2),
				nn.Conv2d(40, 80, 3, 1),
				nn.BatchNorm2d(80, ),
				nn.ReLU(),
				nn.AdaptiveMaxPool2d(output_size=[3, 3]),
			)
		self.fc1 = nn.Linear(80*3*3, 500)
		self.dropout = nn.Dropout(p=0.5, inplace=True)
		self.fc2 = nn.Linear(500, num_classes)

	def forward(self, x):
		x = self.conv(x)
		x = x.view(-1, 80*3*3)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x