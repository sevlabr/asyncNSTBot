# За основу взята модель с семинара DLS
# https://colab.research.google.com/drive/1-X4Q3LkPBLZrQZuLoBj4uA7xP0Hpj8mU

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

import asyncio

from time import time


class ContentLoss(nn.Module):

        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            # we 'detach' the target content from the tree used
            # to dynamically compute the gradient: this is a stated value,
            # not a variable. Otherwise the forward method of the criterion
            # will throw an error.
            self.target = target.detach() #это константа. Убираем ее из дерева вычеслений
            self.loss = F.mse_loss(self.target, self.target) #to initialize with something

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input


def gram_matrix(input):
        batch_size, h, w, f_map_num = input.size()  # batch size(=1)
        # b=number of feature maps
        # (h,w)=dimensions of a feature map (N=h*w)

        features = input.view(batch_size * h, w * f_map_num)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(batch_size * h * w * f_map_num)


class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()
            self.loss = F.mse_loss(self.target, self.target) # to initialize with something

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input


class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std


class Model:

    def __init__(self, imsize):
        self.imsize = imsize
        self.loader = transforms.Compose([
            transforms.Resize(imsize),
            transforms.CenterCrop(imsize),
            transforms.ToTensor()
        ])
        self.unloader = transforms.ToPILImage()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        # self.cnn = models.vgg19(pretrained=True, progress=False).features.to(self.device).eval()
        # self.content_img = None
        # self.style_img = None

    async def image_loader(self, image_name):
        image = Image.open(image_name)
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    async def imshow(self, tensor, title=None):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = self.unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    # def load_images(self, content_path, style_path):
    #     self.content_img = image_loader(content_path)
    #     self.style_img = image_loader(style_path)

    async def get_style_model_and_losses(self, content_path, style_path):
        content_img = await self.image_loader(content_path)
        style_img = await self.image_loader(style_path)

        cnn = copy.deepcopy(self.cnn)

        normalization = Normalization(
            self.cnn_normalization_mean,
            self.cnn_normalization_std
        ).to(self.device)
        
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                # Переопределим relu уровень
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            
            model.add_module(name, layer)

            if name in self.content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
        
        # now we trim off the layers after the last content and style losses
        # выбрасываем все уровни после последенего style loss или content loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        
        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        # добоваляет содержимое тензора катринки в список изменяемых оптимизатором параметров
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    async def run_style_transfer(self, content_path, style_path, to_img=False, time_lim=15*60,
                                 num_steps=500, style_weight=100000, content_weight=1, callback=None):

        """Run the style transfer."""
        tic = time()
        print('Building the style transfer model...')
        model, style_losses, content_losses = await self.get_style_model_and_losses(
            content_path, style_path
        )
        input_img = await self.image_loader(content_path)
        optimizer = self.get_input_optimizer(input_img)

        print('Optimizing...')
        run = [0]
        # Внизу костыль, но зато теперь асинхронно работает :)
        # Время ответа другим пользователям сильно зависит от
        # разрешения обрабатываемой в текущий момент картинки,
        # так как от него зависит время итерации цикла.
        # Тем не менее, бот способен отвечать во время переноса стиля.
        # while run[0] <= num_steps:

        async def gen(i):
            while True:
                yield i
                await asyncio.sleep(1)

        async for step in gen(0):
            step = run[0]
            toc = time()
            if step >= num_steps or toc - tic >= time_lim:
            # if step >= num_steps:
                # print('breaking...\n' + f'{step}')
                break

            def closure():
                # correct the values 
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                #взвешивание ошибки
                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
                    # if run[0] % 100 == 0 and callback is not None:
                    #     await callback.message.answer("Work is in progress...")

                return style_score + content_score

            optimizer.step(closure)

            if callback is not None and run[0] % 100 == 0:
                await callback.message.answer("Style transfer is in progress...")

        # a last correction...
        input_img.data.clamp_(0, 1)
        if to_img:
            input_img = input_img.cpu().clone().squeeze(0)
            input_img = transforms.ToPILImage()(input_img)

        return input_img


async def main():
    NST_model = Model(512)

    output = await NST_model.run_style_transfer("lisa.jpg", "picasso.jpg", num_steps=300)

    plt.figure()
    await NST_model.imshow(output)
    plt.show()

if __name__ == '__main__':
    asyncio.run(main())
