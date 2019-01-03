# basic packages
import os
import os.path as osp
# third-party packages
import pyprind
import glog as log
# pytorch related packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
import argparse
# Model Definition
from model import Net

# Training Part
# Please fill the training part based on the given model/dataloader/optimizer/criterion
def train(args, model, train_loader, optimizer, criterion, epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(
				pyprind.prog_bar(train_loader,
				title="[Epoch {}: Training]".format(epoch),
				width=40,
				)):
                def one_iteration(model, data, target, criterion):
                    '''
                     Please fill the training iteration with given components:

                      model: our provided convolutional neural network
                      data: Chinese Character Images
                      target: category of the images
                      criterion: the loss function
                    '''
                    ##############################
                    print("Please fill the forward iteration.")
                    ##############################

                optimizer.zero_grad()
                one_iteration(model, data, target, criterion)
                optimizer.step()

# Testing Part
def test(args, model, test_loader, epoch):
	model.eval()
	correct = 0
	with torch.no_grad():
		for (data, target) in test_loader:
			output = model(data)
			pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	log.info('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
		correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--epochs', type=int, default=15, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
						help='SGD momentum (default: 0.9)')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--resume', type=str, default=None, help="Model Path.")
	parser.add_argument('--model-name', type=str, default='char_cnn.pt',
						help='Trained model name (defaut: char_cnn.pt).')
	args = parser.parse_args()
	torch.manual_seed(args.seed)

    ## Fill the data directory: [train] and [test] should be at this path:
	data_dir = 'path/to/your'
	# We randomly sample the [image, target] pairs,
	# Then use the pairs to train the model
	trainset = datasets.ImageFolder(
		osp.join(data_dir, 'train'),
		transform=T.Compose([
			# padding the input image
			T.Resize([96, 96]),
			# T.RandomCrop(96, padding=5),
			# T.RandomAffine(degrees=20, ),
			T.ToTensor(),
			T.Normalize(mean=[0.5, 0.5, 0.5,], std=[1., 1., 1.])
			]),
		)
	num_classes = len(trainset.classes)
	valset = datasets.ImageFolder(
		osp.join(data_dir, 'test'),
		transform=T.Compose([
			T.Resize([96, 96]),
			T.ToTensor(),
			T.Normalize(mean=[0.5, 0.5, 0.5,], std=[1., 1., 1.]),
			]),
		)
	assert num_classes == len(valset.classes), "Categories mismatch for train[{}] and validation[{}].".format(num_classes, len(valset.classes))
	# wrap the train/test set into the Dataloader to load the data in batches
	train_loader = torch.utils.data.DataLoader(
		trainset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=4,
		)
	test_loader = torch.utils.data.DataLoader(
		valset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=4,
		)
	# get the model definition
	model = Net(num_classes=num_classes)
	if args.resume:
		model.load_state_dict(torch.load(args.resume))
	lr = args.lr
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
				momentum=args.momentum,
				weight_decay=0.001,
				nesterov=True,)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(1, args.epochs + 1):
		if epoch == int(args.epochs * 0.8): #or epoch == int(args.epochs * 0.8):
			lr = lr * 0.1
			print("[Learning Rate] {}".format(lr))
			for param_group in optimizer.param_groups:
				param_group["lr"] = lr
		train(args, model, train_loader, optimizer, criterion, epoch)
		test(args, model, test_loader, epoch=epoch)

	# saving the trained model and category names
	result = {
		'state_dict': model.state_dict(),
		'classes': trainset.classes,
		'class_to_idx': trainset.class_to_idx,
	}
	torch.save(result, "char_cnn.pt")
	print("Trained Model saved to: {}".format('./char_cnn.pt'))

if __name__ == '__main__':
	main()
