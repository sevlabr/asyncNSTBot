from typing import Union

from aiogram.types import Message, CallbackQuery
from loader import dp
from aiogram.dispatcher.filters import Command

from keyboards import menu_cd, modes_keyboard, photos_request_keyboard, \
    photos_receiver_keyboard, GANStylesKeyboardMaker, quality_decider_keyboard, \
    custom_quality_receiver_keyboard, ask_for_transfer_keyboard

import os.path

from transfer_functions import perform_nst, perform_gan_transfer


HELP_TEXT = "I can work in 2 modes:\n1)NST\n2)GAN\n\nNST requires two pictures. The first is "\
            + "style-picture — the picture from which you would like to transfer style. The second is "\
            + "content-picture — the picture on which you want to transfer style from style-picture.\n\n"\
            + "GAN requires only one picture — content-picture on which you want to transfer style. "\
            + "Style-pictures are set in advance. You will be asked to choose one of given styles.\n\n"\
            + "There is also one more quite significant difference between NST and GAN. GAN works much "\
            + "faster and gives more confident, so to say, results. NST, on the other hand, works "\
            + "with each style-picture more individually, which may lead to more satisfying results. "\
            + "But be careful! The bigger content-picture you send in NST mode the bigger the chance "\
            + "that I will overdo (or underdo) style transfer and you will get strange picture which "\
            + "you don't quite expect. The same can be said about too complex style-pictures.\n\n"\
            + "To proceed write /menu"


# FOLDER_PATH = "Style images for GAN\\"
FOLDER_PATH = "Style images for GAN/"
MAIN_TEXT = "Select style image.\nIf your choice is current photo, click Select button. Otherwise push buttons " \
            + "with names of other pictures to see them. 21 styles are currently available"


class GANStylesDemonstrator(object):
    def __init__(self, photo_name, markup_maker):
        self.photo_name = photo_name
        self.markup_maker = markup_maker

    async def show(self, callback: CallbackQuery, **kwargs):
        markup = await self.markup_maker()
        with open(FOLDER_PATH + self.photo_name, 'rb') as photo:
            await callback.message.reply_photo(
                photo=photo,
                caption=MAIN_TEXT,
                reply_markup=markup,
                reply=False
            )


@dp.message_handler(commands=['help'])
async def send_help(message: Message):
    await message.answer(HELP_TEXT)


@dp.message_handler(commands=['start'])
async def send_start(message: Message):
    await message.answer("Hello!\n\n" + HELP_TEXT)


@dp.message_handler(Command("menu"))
async def show_menu(message: Message):
    await show_modes(message)


@dp.message_handler(content_types=['document'])
async def receive_docs(message):
    # await message.answer("Для переноса стиля необходимо послать фото, а не документ, содержащий фото!\n/help")
    await message.answer("You should send photo but not a document which contains photo!")


@dp.message_handler(content_types=['photo'])
async def receive_photos(message):
    user_id = str(message.chat.id)
    # path = "Current image type\\" + user_id + ".txt"
    path = "Current image type/" + user_id + ".txt"
    if os.path.exists(path):
        with open(path, 'r') as img_path_file:
            img_path = img_path_file.read()
            await message.photo[-1].download(img_path)
            await message.answer("Image saved successfully!")
    else:
        await message.answer("To send photos use /menu and follow further instructions!")


@dp.message_handler(Command("quality"))
async def set_custom_quality(message):
    user_id = str(message.chat.id)
    # path = "Custom image size\\" + user_id + ".txt"
    path = "Custom image size/" + user_id + ".txt"
    with open(path, 'w') as quality_log:
        quality_log.write(message.text[8:])
    await message.answer("Custom size: " + message.text[8:] + " saved successfully!")


async def show_modes(message: Union[CallbackQuery, Message], **kwargs):
    # Клавиатуру формируем с помощью следующей функции
    markup = await modes_keyboard()

    # Проверяем, что за тип апдейта. Если Message - отправляем новое сообщение
    if isinstance(message, Message):
        await message.answer("Choose mode", reply_markup=markup)

    # Если CallbackQuery - изменяем это сообщение
    elif isinstance(message, CallbackQuery):
        call = message
        # await call.message.edit_reply_markup(markup)
        if call.message.text is not None:
            await call.message.edit_text(text="Choose mode", reply_markup=markup)
        else:
            await call.message.answer("Choose mode", reply_markup=markup)


async def photos_request(callback: CallbackQuery, mode, **kwargs):
    markup = await photos_request_keyboard(mode)
    text = "Choose the type of a photo you are going to upload. Then upload the photo.\n" \
           + "When all necessary photos will be uploaded click the Next step button"
    if callback.message.text is not None:
        await callback.message.edit_text(text=text, reply_markup=markup)
    else:
        await callback.message.answer(text=text, reply_markup=markup)


async def photos_receiver(callback: CallbackQuery, mode, photo_type, **kwargs):
    markup = await photos_receiver_keyboard(mode)
    starting_text = "Ok, send the photo which will be used as "
    ending_text = "\nWhen you are done click the upper button"
    user_id = str(callback.message.chat.id)
    if photo_type == "content":
        # img_path = "Content images for NST\\" + user_id + ".jpg"
        img_path = "Content images for NST/" + user_id + ".jpg"
        await callback.message.edit_text(text=starting_text + "content for NST." + ending_text,
                                         reply_markup=markup)
    elif photo_type == "style":
        # img_path = "Style images for NST\\" + user_id + ".jpg"
        img_path = "Style images for NST/" + user_id + ".jpg"
        await callback.message.edit_text(text=starting_text + "style for NST." + ending_text,
                                         reply_markup=markup)
    elif photo_type == "GANcontent":
        # img_path = "Content images for GAN\\" + user_id + ".jpg"
        img_path = "Content images for GAN/" + user_id + ".jpg"
        await callback.message.edit_text(text=starting_text + "content for GAN." + ending_text,
                                         reply_markup=markup)

    # with open("Current image type\\" + user_id + ".txt", 'w') as path:
    with open("Current image type/" + user_id + ".txt", 'w') as path:
        path.write(img_path)


async def quality_decider(callback: CallbackQuery, mode, chosen_photo, prev_lvl, **kwargs):
    markup = await quality_decider_keyboard(mode, chosen_photo, prev_lvl)
    text = "Ok, now choose quality of an output photo.\n"
    if mode == "gan":
        text += "In GAN mode you choose horizontal resolution. The algorithm itself will calculate the desired "\
                + "vertical resolution, keeping the aspect ratio. If you click Initial, the output picture will "\
                + "have exactly the same shape as the photo you've chosen as content-picture for GAN.\n"
    elif mode == "nst":
        text += "In NST mode the algorithm cuts pictures to make them square shaped. So here you choose size of "\
                + "square picture that you will get after style transfer ends. If you click Initial, the algorithm "\
                + "will choose the lowest value of sizes of pictures that you've chosen as style/content-pictures for "\
                + "NST.\n"
    text += "If you want to specify your own resolution click Custom.\nYou can also choose one of preset values:\n"\
            + "Low is 128p, Medium — 256p, High — 512p."
    await callback.message.answer(text=text,
                                  reply_markup=markup)


async def custom_quality_receiver(callback: CallbackQuery, mode, chosen_photo, prev_lvl, **kwargs):
    markup = await custom_quality_receiver_keyboard(mode, chosen_photo, prev_lvl)
    text = "Here you choose your own quality.\nWrite '/quality X', where X is the desired size. "\
           + "After I receive it, click Continue to proceed. Note, that if you choose higher resolution "\
           + "than algorithm would do itself (Initial button), the resulting quality may not satisfy you."
    await callback.message.edit_text(text=text,
                                     reply_markup=markup)


async def ask_for_transfer(callback: CallbackQuery, mode, chosen_photo, output_size, prev_lvl, **kwargs):
    markup = await ask_for_transfer_keyboard(mode, chosen_photo, output_size, prev_lvl)
    text = "Output size: " + output_size + \
           "\nNote, that duration of the algorithm highly depends on the output resolution.\nDo you want to continue?"
    await callback.message.edit_text(text=text,
                                     reply_markup=markup)


async def transfer(callback: CallbackQuery, mode, chosen_photo, output_size, prev_lvl, **kwargs):
    await callback.message.edit_text(text="Style transfer is in progress. It may take a few minutes (up to 10 - 20)")
    print(mode, chosen_photo, output_size, prev_lvl)
    if mode == "nst":
        await perform_nst(callback, mode, output_size)
    elif mode == "gan":
        await perform_gan_transfer(callback, mode, chosen_photo, output_size)


# async def gan_styles_0(callback: CallbackQuery, **kwargs):
#     markup = await gan_styles_keyboard_0()
#     with open(FOLDER_PATH + "candy.jpg", 'rb') as photo:
#         await callback.message.reply_photo(
#             photo=photo,
#             caption=MAIN_TEXT,
#             reply_markup=markup,
#             reply=False,
#         )


@dp.callback_query_handler(menu_cd.filter())
async def navigate(call: CallbackQuery, callback_data: dict):
    """
    :param call: Тип объекта CallbackQuery, который прилетает в хендлер
    :param callback_data: Словарь с данными, которые хранятся в нажатой кнопке
    """

    # Получаем текущий уровень меню, который запросил пользователь
    current_level = callback_data.get("level")

    # Получаем режим работы, который выбрал пользователь
    mode = callback_data.get("mode")

    photo_type = callback_data.get("photo_type")

    chosen_photo = callback_data.get("chosen_photo")

    output_size = callback_data.get("output_size")

    prev_lvl = callback_data.get("prev_lvl")

    # Прописываем "уровни" в которых будут отправляться новые кнопки пользователю
    levels = {
        "0": show_modes,  # Показываем доступные режимы
        "1": photos_request,
        "2": photos_receiver,
        "3": GANStylesDemonstrator(
                "candy.jpg",
                GANStylesKeyboardMaker(
                    lvl=3, prev_lvl=23, next_lvl=4, next_2_lvl=5,
                    l_text="✅ Candy", m_text="Composition VII", r_text="Escher sphere",
                    chosen_photo="candy.jpg"
                ).make
            ).show,
        "4": GANStylesDemonstrator(
                "composition_vii.jpg",
                GANStylesKeyboardMaker(
                    lvl=4, prev_lvl=3, next_lvl=5, next_2_lvl=6,
                    l_text="✅ Composition VII", m_text="Escher sphere", r_text="Feathers",
                    chosen_photo="composition_vii.jpg"
                ).make
            ).show,
        "5": GANStylesDemonstrator(
                "escher_sphere.jpg",
                GANStylesKeyboardMaker(
                    lvl=5, prev_lvl=4, next_lvl=6, next_2_lvl=7,
                    l_text="✅ Escher sphere", m_text="Feathers", r_text="Frida\nKahlo",
                    chosen_photo="escher_sphere.jpg"
                ).make
            ).show,
        "6": GANStylesDemonstrator(
                "feathers.jpg",
                GANStylesKeyboardMaker(
                    lvl=6, prev_lvl=5, next_lvl=7, next_2_lvl=8,
                    l_text="✅ Feathers", m_text="Frida\nKahlo", r_text="La Muse",
                    chosen_photo="feathers.jpg"
                ).make
            ).show,
        "7": GANStylesDemonstrator(
                "frida_kahlo.jpg",
                GANStylesKeyboardMaker(
                    lvl=7, prev_lvl=6, next_lvl=8, next_2_lvl=9,
                    l_text="✅ Frida\nKahlo", m_text="La Muse", r_text="Mosaic",
                    chosen_photo="frida_kahlo.jpg"
                ).make
            ).show,
        "8": GANStylesDemonstrator(
                "la_muse.jpg",
                GANStylesKeyboardMaker(
                    lvl=8, prev_lvl=7, next_lvl=9, next_2_lvl=10,
                    l_text="✅ La Muse", m_text="Mosaic", r_text="Ancient\nmosaic",
                    chosen_photo="la_muse.jpg"
                ).make
            ).show,
        "9": GANStylesDemonstrator(
                "mosaic.jpg",
                GANStylesKeyboardMaker(
                    lvl=9, prev_lvl=8, next_lvl=10, next_2_lvl=11,
                    l_text="✅ Mosaic", m_text="Ancient\nmosaic", r_text="Pencil",
                    chosen_photo="mosaic.jpg"
                ).make
            ).show,
        "10": GANStylesDemonstrator(
                "mosaic_ducks_massimo.jpg",
                GANStylesKeyboardMaker(
                    lvl=10, prev_lvl=9, next_lvl=11, next_2_lvl=12,
                    l_text="✅ Ancient\nmosaic", m_text="Pencil", r_text="Picasso\nself-portrait",
                    chosen_photo="mosaic_ducks_massimo.jpg"
                ).make
            ).show,
        "11": GANStylesDemonstrator(
                "pencil.jpg",
                GANStylesKeyboardMaker(
                    lvl=11, prev_lvl=10, next_lvl=12, next_2_lvl=13,
                    l_text="✅ Pencil", m_text="Picasso\nself-portrait", r_text="Afremov\nRain princess",
                    chosen_photo="pencil.jpg"
                ).make
            ).show,
        "12": GANStylesDemonstrator(
                "picasso_selfport1907.jpg",
                GANStylesKeyboardMaker(
                    lvl=12, prev_lvl=11, next_lvl=13, next_2_lvl=14,
                    l_text="✅ Picasso\nself-portrait", m_text="Afremov\nRain princess",
                    r_text="Robert Delaunay\nportrait",
                    chosen_photo="picasso_selfport1907.jpg"
                ).make
            ).show,
        "13": GANStylesDemonstrator(
                "rain_princess.jpg",
                GANStylesKeyboardMaker(
                    lvl=13, prev_lvl=12, next_lvl=14, next_2_lvl=15,
                    l_text="✅ Afremov\nRain princess", m_text="Robert Delaunay\nportrait",
                    r_text="Picasso\nSeated Nude",
                    chosen_photo="rain_princess.jpg"
                ).make
            ).show,
        "14": GANStylesDemonstrator(
                "RDelaunayPortrait.jpg",
                GANStylesKeyboardMaker(
                    lvl=14, prev_lvl=13, next_lvl=15, next_2_lvl=16,
                    l_text="✅ Robert Delaunay\nportrait", m_text="Picasso\nSeated Nude",
                    r_text="Turner\nShipwreck",
                    chosen_photo="RDelaunayPortrait.jpg"
                ).make
            ).show,
        "15": GANStylesDemonstrator(
                "seated-nude.jpg",
                GANStylesKeyboardMaker(
                    lvl=15, prev_lvl=14, next_lvl=16, next_2_lvl=17,
                    l_text="✅ Picasso\nSeated Nude", m_text="Turner\nShipwreck",
                    r_text="Starry night",
                    chosen_photo="seated-nude.jpg"
                ).make
            ).show,
        "16": GANStylesDemonstrator(
                "shipwreck.jpg",
                GANStylesKeyboardMaker(
                    lvl=16, prev_lvl=15, next_lvl=17, next_2_lvl=18,
                    l_text="✅ Turner\nShipwreck", m_text="Starry night",
                    r_text="Stars",
                    chosen_photo="shipwreck.jpg"
                ).make
            ).show,
        "17": GANStylesDemonstrator(
                "starry_night.jpg",
                GANStylesKeyboardMaker(
                    lvl=17, prev_lvl=16, next_lvl=18, next_2_lvl=19,
                    l_text="✅ Starry night", m_text="Stars",
                    r_text="Jackson Pollock\nGray Painting",
                    chosen_photo="starry_night.jpg"
                ).make
            ).show,
        "18": GANStylesDemonstrator(
                "stars2.jpg",
                GANStylesKeyboardMaker(
                    lvl=18, prev_lvl=17, next_lvl=19, next_2_lvl=20,
                    l_text="✅ Stars", m_text="Jackson Pollock\nGray Painting",
                    r_text="Munch\nThe Scream",
                    chosen_photo="stars2.jpg"
                ).make
            ).show,
        "19": GANStylesDemonstrator(
                "strip.jpg",
                GANStylesKeyboardMaker(
                    lvl=19, prev_lvl=18, next_lvl=20, next_2_lvl=21,
                    l_text="✅ Jackson Pollock\nGray Painting", m_text="Munch\nThe Scream",
                    r_text="Udnie",
                    chosen_photo="strip.jpg"
                ).make
            ).show,
        "20": GANStylesDemonstrator(
                "the_scream.jpg",
                GANStylesKeyboardMaker(
                    lvl=20, prev_lvl=19, next_lvl=21, next_2_lvl=22,
                    l_text="✅ Munch\nThe Scream", m_text="Udnie",
                    r_text="The Wave",
                    chosen_photo="the_scream.jpg"
                ).make
            ).show,
        "21": GANStylesDemonstrator(
                "udnie.jpg",
                GANStylesKeyboardMaker(
                    lvl=21, prev_lvl=20, next_lvl=22, next_2_lvl=23,
                    l_text="✅ Udnie", m_text="The Wave",
                    r_text="Matisse\nWoman with a Hat",
                    chosen_photo="udnie.jpg"
                ).make
            ).show,
        "22": GANStylesDemonstrator(
                "wave.jpg",
                GANStylesKeyboardMaker(
                    lvl=22, prev_lvl=21, next_lvl=23, next_2_lvl=3,
                    l_text="✅ The Wave", m_text="Matisse\nWoman with a Hat",
                    r_text="Candy",
                    chosen_photo="wave.jpg"
                ).make
            ).show,
        "23": GANStylesDemonstrator(
                "woman-with-hat-matisse.jpg",
                GANStylesKeyboardMaker(
                    lvl=23, prev_lvl=22, next_lvl=3, next_2_lvl=4,
                    l_text="✅ Matisse\nWoman with a Hat", m_text="Candy",
                    r_text="Composition VII",
                    chosen_photo="woman-with-hat-matisse.jpg"
                ).make
            ).show,
        "24": quality_decider,
        "25": custom_quality_receiver,
        "26": ask_for_transfer,
        "27": transfer
    }

    # Забираем нужную функцию для выбранного уровня
    current_level_function = levels[current_level]

    # Выполняем нужную функцию и передаем туда параметры, полученные из кнопки
    await current_level_function(
        call,
        mode=mode,
        photo_type=photo_type,
        chosen_photo=chosen_photo,
        output_size=output_size,
        prev_lvl=prev_lvl
    )
