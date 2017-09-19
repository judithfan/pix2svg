"""Neural Style Transfer. This is almost copy-pasta
from Alexis Jacq's PyTorch tutorial:
https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss


class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss


def get_style_model_and_losses(cnn, style_img, content_img,
                               style_weight=1000, content_weight=1,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default,
                               use_cuda=False):
    cnn = copy.deepcopy(cnn)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential()  # the new Sequential module network
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    if use_cuda:
        model = model.cuda()
        gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***

    return model, style_losses, content_losses


def get_input_param_optimizer(input_img, init_lr=0.1):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param], lr=init_lr)
    return input_param, optimizer


def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300,
                       style_weight=1000, content_weight=1, use_cuda=False,
                       init_lr=0.1, anneal_freq=100):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        style_img, content_img, style_weight, content_weight, use_cuda=use_cuda)
    input_param, optimizer = get_input_param_optimizer(input_img, init_lr=init_lr)

    def adjust_learning_rate(optimizer, epoch):
        lr = init_lr * (0.1** (epoch // anneal_freq))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if epoch % anneal_freq == 0:
            print('Learning rate set to {}'.format(lr))

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_param.data.clamp_(0, 1)
            optimizer.zero_grad()
            #import pdb; pdb.set_trace()
            model(input_param)
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.data[0], content_score.data[0]))
                print()
            return style_score + content_score
        
        optimizer.step(closure)
        
    adjust_learning_rate(optimizer, run[0])

    # a last correction...
    input_param.data.clamp_(0, 1)

    return input_param.data


def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image


def imshow(tensor, imsize, title=None):
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(3, imsize, imsize)  # remove the fake batch dimension
    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='neural style')
    parser.add_argument('--n_iters', type=int, default=300)
    parser.add_argument('--style_weight', type=int, default=1000)
    parser.add_argument('--content_weight', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./outputs/output.pt')
    parser.add_argument('--init_lr', type=float, default=1)
    parser.add_argument('--anneal_freq', type=int, default=100)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--style_img', type=str, default = "./images/sketch.jpg")
    parser.add_argument('--content_img', type=str, default = "./images/dancing.jpg")    
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    # desired size of the output image
    imsize = 512 if use_cuda else 128  # use small size if no gpu

    loader = transforms.Compose([
        transforms.Scale(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    style_img = image_loader(args.style_img).type(dtype)
    content_img = image_loader(args.content_img).type(dtype)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    cnn = models.vgg19(pretrained=True).features

    # move it to the GPU if possible:
    if use_cuda:
        cnn = cnn.cuda(args.cuda_device)

    # input_img = content_img.clone()
    input_img = Variable(torch.rand(style_img.size())).type(dtype)
    output = run_style_transfer(cnn, content_img, style_img, input_img, use_cuda=use_cuda,
                                num_steps=args.n_iters, style_weight=args.style_weight,
                                content_weight=args.content_weight, init_lr=args.init_lr,
                                anneal_freq=args.anneal_freq)
    torch.save(output, args.save_path)
