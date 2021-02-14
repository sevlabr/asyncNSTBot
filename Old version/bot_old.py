import logging

from aiogram import Bot, Dispatcher, executor, types

from models.model_nst import Model

from models.model_gan import Net
from models.functions_gan import *

from torch.autograd import Variable

from PIL import Image

import os

import asyncio

from functools import partial

API_TOKEN = 'YOUR BOT TOKEN HERE'

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_start(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/help` command
    """
    await message.reply("Привет!\n Я могу работать в двух режимах:\n1) NST\n2) GAN.\n\nНемного о каждом из них.\n\n"
                        + "Neural style transfer (NST).\nОтправь мне две картинки: картинку, на которую нужно перенести "
                        + "стиль, её называют 'Content', и картинку, которая определяет стиль (содержит его), "
                        + "её называют 'Style'. Обычно вторая картинка — это какая-то картина известного художника, "
                        + "но не обязательно. Первая картинка может быть любой. Я перенесу на Content-картинку стиль "
                        + "Style-картинки.\nОбрати внимание на то, что алгоритм приводит картинку к квадратной, "
                        + "поэтому лучше не посылать вытянутые картинки, результат может тебе не понравиться. "
                        + "По умолчанию разрешение картинки с перенесенным стилем равно минимальному из "
                        + "четырех чисел HxW, которые являются соответственно высотой и шириной исходных картинок. "
                        + "Если хочется задать своё разрешение, его можно указать перед началом работы алгоритма.\n"
                        + "Чтобы начать, напиши '/nst'.\n\n"
                        + "Generative adversarial network (GAN).\nСначала отправь мне картинку, на которую ты хотел бы "
                        + "перенести стиль (назовем её 'GANContent'). Затем нужно выбрать один из предложенных стилей. "
                        + "Чтобы посмотреть, с какими стилями я умею работать, напиши команду '/ganstyles'. "
                        + "Всего я знаю 21 стиль. При выполнении этой команды я пошлю все картинки-стили с "
                        + "названиями этих стилей. После этого останется только отправить мне команду и я начну "
                        + "переносить стиль!\nТакже доступна экспериментальная опция выбора своей картинки-стиля "
                        + "'GANStyle', но она может выдавать абсолютно неожиданные результаты. Особенно, если твоя "
                        + "картинка-стиль сильно отличается от всех тех картинок-стилей, которые знаю я (это те, "
                        + "которые передаются по команде '/ganstyles').\nЧтобы начать, напиши '/gan'.")


@dp.message_handler(commands=['gan'])
async def send_welcome(message: types.Message):
    await message.reply("Итак, давай начнём!\nТебе нужно послать мне картинку, на которую ты хотел бы "
                        + "перенести стиль. В подписи к ней отправь слово 'GANContent'.\nЧтобы я начал переносить "
                        + "стиль, введи команду вида "
                        + "'/gantransfer StyleName X', где StyleName — название картинки-стиля (можно посмотреть "
                        + "с помощью '/ganstyles'), а X — разрешение получающейся картинки по ширине (алгоритм "
                        + "сам вычислит нужное значение разрешения по высоте, сохраняя пропорции). Если не указать X, "
                        + "написав '/gantransfer StyleName', то картинка на выходе будет иметь такие же размеры как "
                        + "у входной картинки.\nНаконец, если тебе хочется попробовать экспериментальный режим, "
                        + "то ты можешь отправить мне свою картинку-стиль с подписью 'GANStyle'. Тогда в команде "
                        + "'/gantransfer' в качестве StyleName нужно будет указать 'custom'.\nОбрати внимание на то, "
                        + "что, во-первых, чем больше разрешение выходной картинки, тем больше времени придется "
                        + "ждать результата, а во-вторых, в случае собственной картинки-стиля (режим 'custom') "
                        + "результирующий стиль, скорее всего, не будет похож на тот, что имеет твоя картинка.")


@dp.message_handler(commands=['nst'])
async def send_welcome(message: types.Message):
    await message.reply("Итак, давай начнём!\nТебе нужно послать мне две картинки:\n"
                        + "Первая — та, на которую будет переноситься стиль. "
                        + "В подписи к ней отправь слово 'Content'.\nВторая — картинка со стилем. "
                        + "В подписи оставь 'Style'.\nКак только я получу обе картинки, напиши '/nsttransfer', "
                        + "и я начну переносить стиль!\nЕсли хочешь указать желаемое разрешение "
                        + "картинки на выходе, напиши '/nsttransfer X', где X — целое строго положительное число. "
                        + "Желательно, чтобы оно было не больше разрешения посылаемых картинок.\n"
                        + "Например, '/nsttransfer 512' вернёт картинку с разрешением 512x512.\n"
                        + "Время работы алгоритма сильно зависит от разрешения итоговой картинки!")


@dp.message_handler(content_types=['document'])
async def receive_docs(message):
    await message.answer("Для переноса стиля необходимо послать фото, а не документ, содержащий фото!\n/help")


@dp.message_handler(content_types=['photo'])
async def receive_photos(message):
    """
    Can't be used in group chats. Private chats only
    """
    user_id = message.chat.id
    if message.caption == "Content":
        await message.photo[-1].download('content' + str(user_id) + '.jpg')
    elif message.caption == "Style":
        await message.photo[-1].download('style' + str(user_id) + '.jpg')
    elif message.caption == "GANContent":
        await message.photo[-1].download('gan_content' + str(user_id) + '.jpg')
    elif message.caption == "GANStyle":
        await message.photo[-1].download('gan_style' + str(user_id) + '.jpg')
    else:
        await message.answer("Обязательно нужно написать, какая именно картинка отправляется!\nContent / Style\n/help")


async def get_resolution(content_pth, style_pth):
    content_width, content_height = Image.open(content_pth).size
    style_width, style_height = Image.open(style_pth).size
    proportions = sorted([content_width, content_height, style_width, style_height])
    return proportions[0]


@dp.message_handler(commands=['nsttransfer'])
async def perform_nst(message):
    """
    Can't be used in group chats. Private chats only
    """
    await message.answer("Начинаю перенос стиля!\nЭто может занять ~ 15 минут!")
    usr_id = message.chat.id
    content_pth = 'content' + str(usr_id) + '.jpg'
    style_pth = 'style' + str(usr_id) + '.jpg'
    resulting_res = await get_resolution(content_pth, style_pth)
    if len(message.text) != 12:
        resulting_res = int(message.text[13:])
    nst_model = Model(resulting_res)
    output = await nst_model.run_style_transfer(
        content_pth,
        style_pth,
        to_img=True,
        style_weight=100000
    )
    output_path = 'result' + str(usr_id) + '.jpg'
    output.save(output_path)
    with open(output_path, 'rb') as photo:
        await bot.send_photo(chat_id=usr_id, photo=photo)


@dp.message_handler(commands=['ganstyles'])
async def show_gan_styles(message):
    # img_dir_name = '..\\Style images for GAN'
    img_dir_name = '../Style images for GAN'
    dir_to_show = os.listdir(img_dir_name)
    for item in dir_to_show:
        with open(os.path.join(img_dir_name, item), 'rb') as photo:
            await bot.send_photo(
                chat_id=message.chat.id,
                photo=photo,
                caption=item[:-4]
            )


# Create MSG-Net and load pre-trained weights
async def get_gan_model(ngf=64):
    style_model = Net(ngf=ngf)
    # model_dict = torch.load('..\\21styles.model')
    model_dict = torch.load('../21styles.model')
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
        'gan_result' + str(message.chat.id) + '.jpg',
        False
    )


@dp.message_handler(commands=['gantransfer'])
async def perform_gan_transfer(message):
    await message.answer("Начинаю перенос стиля!\nЭто может занять некоторое время!")
    words = message.text.split()
    style_name = None
    if len(words) >= 2:
        style_name = words[1]
    user_size = None
    if len(words) >= 3:
        user_size = int(words[2])
    style_path = None
    if style_name != 'custom':
        # style_path = '..\\Style images for GAN\\' + style_name + '.jpg'
        style_path = '../Style images for GAN/' + style_name + '.jpg'
    else:
        style_path = 'gan_style' + str(message.chat.id) + '.jpg'
    style_model = await get_gan_model(ngf=128)
    content_image, style = await get_gan_imgs(
        'gan_content' + str(message.chat.id) + '.jpg',
        style_path,
        user_size=user_size
    )
    await gan_transfer(style_model, content_image, style, message)
    with open('gan_result' + str(message.chat.id) + '.jpg', 'rb') as photo:
        await bot.send_photo(chat_id=message.chat.id, photo=photo)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
