from models.model_gan import Net

import numpy as np
import torch

from PIL import Image

from torch.autograd import Variable


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)),
                         Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch


if __name__ == '__main__':
    # Load the images and preprocessing
    content_image = tensor_load_rgbimage('lisa.jpg', size=181,
                                         keep_asp=True).unsqueeze(0)
    style = tensor_load_rgbimage('wave.jpg', size=181).unsqueeze(0)
    style = preprocess_batch(style)

    # Create MSG-Net and load pre-trained weights
    style_model = Net(ngf=128)
    # model_dict = torch.load('..\\21styles.model')
    model_dict = torch.load('../21styles.model')
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)

    # Set the style target and generate outputs
    style_v = Variable(style)
    content_image = Variable(preprocess_batch(content_image))
    style_model.setTarget_sync(style_v)
    output = style_model(content_image)
    tensor_save_bgrimage(output.data[0], 'output.jpg', False)
