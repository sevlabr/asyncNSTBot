from aiogram.types import CallbackQuery

import os.path

from models.model_nst import Model

from models.model_gan import Net
from models.functions_gan import *

from torch.autograd import Variable

import asyncio

from functools import partial

import gc


# CONTENT_PATH_NST = "Content images for NST\\"
# STYLE_PATH_NST = "Style images for NST\\"
CONTENT_PATH_NST = "Content images for NST/"
STYLE_PATH_NST = "Style images for NST/"


async def get_resolution(content_pth, style_pth):
    content_width, content_height = Image.open(content_pth).size
    style_width, style_height = Image.open(style_pth).size
    proportions = sorted([content_width, content_height, style_width, style_height])
    return proportions[0]


async def get_resulting_res(callback: CallbackQuery, mode, output_size, user_id, content_path=None, style_path=None):
    resulting_res = 0
    if output_size == "custom":
        # custom_size_path = "Custom image size\\" + str(user_id) + ".txt"
        custom_size_path = "Custom image size/" + str(user_id) + ".txt"
        if os.path.exists(custom_size_path):
            with open(custom_size_path, 'r') as quality_log:
                resulting_res = int(quality_log.read())
        else:
            await callback.message.answer("You've chosen custom output size but forgot to specify it! Use /menu -> "
                                          + "Custom quality and then write /quality X, where X is the resolution "
                                          + "you want.")

    elif output_size == "initial" and mode == "nst" and content_path and style_path:
        resulting_res = await get_resolution(content_path, style_path)

    elif output_size != "initial" and output_size != "custom":
        resulting_res = int(output_size)

    return resulting_res


async def perform_nst(callback: CallbackQuery, mode, output_size):
    user_id = callback.message.chat.id
    content_path = CONTENT_PATH_NST + str(user_id) + ".jpg"
    style_path = STYLE_PATH_NST + str(user_id) + ".jpg"
    if not os.path.exists(content_path) or not os.path.exists(style_path):
        await callback.message.answer("Looks like you forgot to send photos! (Or at least one of them)\n"
                                      + "To fix it use /menu -> Style/Content photo and then send photos you want "
                                      + "me to use!")

    else:
        resulting_res = await get_resulting_res(callback, mode, output_size, user_id,
                                                content_path=content_path, style_path=style_path)
        if resulting_res > 0:
            nst_model = Model(resulting_res)
            output = await nst_model.run_style_transfer(
                content_path,
                style_path,
                to_img=True,
                # num_steps=200,
                style_weight=100000,
                callback=callback
            )
            # output_path = "Results\\NST" + str(user_id) + ".jpg"
            output_path = "Results/NST" + str(user_id) + ".jpg"
            output.save(output_path)
            with open(output_path, 'rb') as photo:
                await callback.message.answer_photo(photo=photo)

            del output
            del nst_model

        else:
            await callback.message.answer("Looks like something is wrong with output size!\n"
                                          + "Output size: " + output_size + "\nResulting resolution: "
                                          + str(resulting_res) + "\nAborting style transfer...")

    torch.cuda.empty_cache()
    gc.collect()


# Create MSG-Net and load pre-trained weights
async def get_gan_model(ngf=64):
    style_model = Net(ngf=ngf)
    model_dict = torch.load('21styles.model')
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)
    return style_model


async def get_gan_imgs(content_pth, style_pth, user_size=None):
    content_width, content_height = Image.open(content_pth).size
    if user_size is not None:
        content_width = user_size
    content_image = await tensor_load_rgbimage(content_pth, size=content_width,
                                               keep_asp=True)
    content_image = content_image.unsqueeze(0)
    style_image = await tensor_load_rgbimage(style_pth, size=content_width)
    style_image = style_image.unsqueeze(0)
    style_image = await preprocess_batch(style_image)
    return content_image, style_image


# Set the style target and generate outputs
async def gan_transfer(style_model, content_image, style, message):
    style_v = Variable(style)
    content_image = Variable(await preprocess_batch(content_image))
    await style_model.setTarget(style_v)
    # output = await style_model(content_image)
    loop = asyncio.get_running_loop()
    output = await loop.run_in_executor(
        None,
        partial(style_model, content_image)
    )
    await tensor_save_bgrimage(
        output.data[0],
        # 'Results\\GAN' + str(message.chat.id) + '.jpg',
        'Results/GAN' + str(message.chat.id) + '.jpg',
        False
    )

    del style_v
    del output


async def perform_gan_transfer(callback: CallbackQuery, mode, chosen_photo, output_size):
    user_id = callback.message.chat.id
    # style_path = "Style images for GAN\\" + chosen_photo
    # content_path = "Content images for GAN\\" + str(user_id) + ".jpg"
    style_path = "Style images for GAN/" + chosen_photo
    content_path = "Content images for GAN/" + str(user_id) + ".jpg"
    if not os.path.exists(content_path):
        await callback.message.answer("Looks like you forgot to send content photo!\n"
                                      + "To fix it use /menu -> Content photo and then send the photo you want "
                                      + "me to use!")
    else:
        style_model = await get_gan_model(ngf=128)
        if output_size != "initial":
            resulting_res = await get_resulting_res(callback, mode, output_size, user_id)
        else:
            resulting_res = None

        if isinstance(resulting_res, int) and resulting_res > 0 or resulting_res is None:
            content_image, style = await get_gan_imgs(
                content_path,
                style_path,
                user_size=resulting_res,
            )
            await gan_transfer(style_model, content_image, style, callback.message)
            # with open('Results\\GAN' + str(user_id) + '.jpg', 'rb') as photo:
            with open('Results/GAN' + str(user_id) + '.jpg', 'rb') as photo:
                await callback.message.answer_photo(photo=photo)

            del content_image
            del style

        else:
            await callback.message.answer("Looks like something is wrong with output size!\n"
                                          + "Output size: " + output_size + "\nResulting resolution: "
                                          + str(resulting_res) + "\nAborting style transfer...")
        del style_model

    torch.cuda.empty_cache()
    gc.collect()
