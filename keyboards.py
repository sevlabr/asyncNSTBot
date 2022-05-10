from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.callback_data import CallbackData


# Создаем CallbackData-объекты, которые будут нужны для работы с меню
menu_cd = CallbackData("show_menu", "level", "mode", "photo_type", "chosen_photo", "output_size", "prev_lvl")

# С помощью этой функции будем формировать коллбек дату для каждого элемента меню, в зависимости от
# переданных параметров
def make_callback_data(level, mode="0", photo_type="0", chosen_photo="0", output_size="0", prev_lvl=0):
    return menu_cd.new(level=level, mode=mode, photo_type=photo_type,
                       chosen_photo=chosen_photo, output_size=output_size,
                       prev_lvl=prev_lvl)


async def modes_keyboard():
    # Указываем, что текущий уровень меню - 0
    CURRENT_LEVEL = 0

    # Создаем Клавиатуру
    markup = InlineKeyboardMarkup()

    nst_button_text = "NST mode"
    gan_button_text = "GAN mode"

    nst_callback_data = make_callback_data(level=CURRENT_LEVEL + 1, mode="nst")
    gan_callback_data = make_callback_data(level=CURRENT_LEVEL + 1, mode="gan")

    markup.insert(
        InlineKeyboardButton(text=nst_button_text, callback_data=nst_callback_data)
    )
    markup.insert(
        InlineKeyboardButton(text=gan_button_text, callback_data=gan_callback_data)
    )

    return markup


async def photos_request_keyboard(mode):
    # Текущий уровень - 1
    CURRENT_LEVEL = 1
    markup = InlineKeyboardMarkup()

    if mode == "nst":
        style_button_text = "Style photo"
        content_button_text = "Content photo"

        style_callback_data = make_callback_data(level=CURRENT_LEVEL + 1,
                                                 mode="nst", photo_type="style")
        content_callback_data = make_callback_data(level=CURRENT_LEVEL + 1,
                                                   mode="nst", photo_type="content")

        markup.insert(
            InlineKeyboardButton(text=style_button_text, callback_data=style_callback_data)
        )
        markup.insert(
            InlineKeyboardButton(text=content_button_text, callback_data=content_callback_data)
        )

        step = 23

    elif mode == "gan":
        content_button_text = "Content photo"

        content_callback_data = make_callback_data(level=CURRENT_LEVEL + 1,
                                                   mode="gan", photo_type="GANcontent")

        markup.insert(
            InlineKeyboardButton(text=content_button_text, callback_data=content_callback_data)
        )

        step = 2

    next_step_button_text = "Next step"
    back_step_button_text = "Back"

    next_step_callback_data = make_callback_data(level=CURRENT_LEVEL + step, mode=mode)
    back_callback_data = make_callback_data(level=CURRENT_LEVEL - 1)

    markup.row(
        InlineKeyboardButton(
            text=next_step_button_text,
            callback_data=next_step_callback_data
        ),
        InlineKeyboardButton(
            text=back_step_button_text,
            callback_data=back_callback_data
        )
    )

    markup.row(
        InlineKeyboardButton(
            text="Cancel",
            callback_data=make_callback_data(level=0)
        )
    )

    return markup


async def photos_receiver_keyboard(mode):
    # Текущий уровень - 2
    CURRENT_LEVEL = 2
    markup = InlineKeyboardMarkup()

    button_text = "Back to type of photo choosing"
    callback_data = make_callback_data(level=CURRENT_LEVEL - 1, mode=mode)

    markup.row(
        InlineKeyboardButton(
            text=button_text,
            callback_data=callback_data
        ),
    )

    markup.row(
        InlineKeyboardButton(
            text="Cancel",
            callback_data=make_callback_data(level=0)
        )
    )

    return markup


async def quality_decider_keyboard(mode, chosen_photo, prev_lvl):
    CURRENT_LEVEL = 24
    markup = InlineKeyboardMarkup()

    if mode == "nst":
        prev_lvl = 1

    markup.row(
        InlineKeyboardButton(
            text="Custom",
            callback_data=make_callback_data(level=CURRENT_LEVEL + 1, mode=mode, chosen_photo=chosen_photo,
                                             prev_lvl=prev_lvl)
        ),
        InlineKeyboardButton(
            text="Initial",
            callback_data=make_callback_data(level=CURRENT_LEVEL + 2, mode=mode, chosen_photo=chosen_photo,
                                             output_size="initial", prev_lvl=prev_lvl)
        )
    )

    markup.row(
        InlineKeyboardButton(
            text="Low",
            callback_data=make_callback_data(level=CURRENT_LEVEL + 2, mode=mode, chosen_photo=chosen_photo,
                                             output_size="128", prev_lvl=prev_lvl)
        ),
        InlineKeyboardButton(
            text="Medium",
            callback_data=make_callback_data(level=CURRENT_LEVEL + 2, mode=mode, chosen_photo=chosen_photo,
                                             output_size="256", prev_lvl=prev_lvl)
        ),
        InlineKeyboardButton(
            text="High",
            callback_data=make_callback_data(level=CURRENT_LEVEL + 2, mode=mode, chosen_photo=chosen_photo,
                                             output_size="512", prev_lvl=prev_lvl)
        )
    )

    markup.row(
        InlineKeyboardButton(
            text="Cancel",
            callback_data=make_callback_data(level=0)
        ),
        InlineKeyboardButton(
            text="Back",
            callback_data=make_callback_data(level=prev_lvl, mode=mode)
        )
    )

    return markup


async def custom_quality_receiver_keyboard(mode, chosen_photo, prev_lvl):
    CURRENT_LEVEL = 25
    markup = InlineKeyboardMarkup()

    markup.row(
        InlineKeyboardButton(
            text="Continue",
            callback_data=make_callback_data(level=CURRENT_LEVEL + 1, mode=mode,
                                             chosen_photo=chosen_photo, output_size="custom",
                                             prev_lvl=prev_lvl)
        )
    )

    markup.row(
        InlineKeyboardButton(
            text="Cancel",
            callback_data=make_callback_data(level=0)
        ),
        InlineKeyboardButton(
            text="Back",
            callback_data=make_callback_data(level=CURRENT_LEVEL - 1, mode=mode,
                                             chosen_photo=chosen_photo, prev_lvl=prev_lvl)
        )
    )

    return markup


async def ask_for_transfer_keyboard(mode, chosen_photo, output_size, prev_lvl):
    CURRENT_LEVEL = 26
    markup = InlineKeyboardMarkup()

    markup.row(
        InlineKeyboardButton(
            text="Yes, start style transfer",
            callback_data=make_callback_data(level=CURRENT_LEVEL + 1, mode=mode,
                                             chosen_photo=chosen_photo, output_size=output_size,
                                             prev_lvl=prev_lvl)
        )
    )

    markup.row(
        InlineKeyboardButton(
            text="Cancel",
            callback_data=make_callback_data(level=0)
        ),
        InlineKeyboardButton(
            text="Back",
            callback_data=make_callback_data(level=CURRENT_LEVEL - 2, mode=mode,
                                             chosen_photo=chosen_photo, prev_lvl=prev_lvl)
        )
    )

    return markup


class GANStylesKeyboardMaker(object):
    def __init__(self, lvl, prev_lvl, next_lvl, next_2_lvl, chosen_photo):
        self.lvl = lvl
        self.prev_lvl = prev_lvl
        self.next_lvl = next_lvl
        self.next_2_lvl = next_2_lvl
        self.chosen_photo = chosen_photo

    async def make(self):
        CURRENT_LEVEL = self.lvl
        markup = InlineKeyboardMarkup()

        select_callback_data = make_callback_data(level=24, mode="gan",
                                                  chosen_photo=self.chosen_photo,
                                                  prev_lvl=CURRENT_LEVEL)

        markup.row(
            InlineKeyboardButton(text="Previous photo", callback_data=make_callback_data(level=self.prev_lvl)),
            InlineKeyboardButton(text="Next photo", callback_data=make_callback_data(level=self.next_lvl))
        )
        markup.row(
            InlineKeyboardButton(text="Select", callback_data=select_callback_data)
        )
        markup.row(
            InlineKeyboardButton(text="Cancel", callback_data=make_callback_data(level=0)),
            InlineKeyboardButton(text="Back", callback_data=make_callback_data(level=1, mode="gan"))
        )

        return markup


# async def gan_styles_keyboard_0():
#     CURRENT_LEVEL = 3
#     markup = InlineKeyboardMarkup()
#
#     left_button_text = "✅ Candy"
#     middle_button_text = "Composition VII"
#     right_button_text = "Escher sphere"
#
#     left_callback_data = make_callback_data(level=CURRENT_LEVEL)
#     middle_callback_data = make_callback_data(level=CURRENT_LEVEL + 1)
#     right_callback_data = make_callback_data(level=CURRENT_LEVEL + 2)
#
#     markup.row(
#         InlineKeyboardButton(text=left_button_text, callback_data=left_callback_data),
#         InlineKeyboardButton(text=middle_button_text, callback_data=middle_callback_data),
#         InlineKeyboardButton(text=right_button_text, callback_data=right_callback_data)
#     )
#     markup.row(
#         InlineKeyboardButton(text="Select", callback_data=make_callback_data(level=22)),
#         InlineKeyboardButton(text="Previous photo", callback_data=make_callback_data(level=21))
#     )
#     markup.row(
#         InlineKeyboardButton(text="Cancel", callback_data=make_callback_data(level=0)),
#         InlineKeyboardButton(text="Back", callback_data=make_callback_data(level=1, mode="gan"))
#     )
#
#     return markup
