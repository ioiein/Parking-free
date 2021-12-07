from telegram import Update, InputMediaPhoto, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext


def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    buttons = [[KeyboardButton("Show")]]
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
        reply_markup=ReplyKeyboardMarkup(buttons, resize_keyboard=True),
    )


def messageHandler(update: Update, context: CallbackContext):
    #if update.effective_chat.username not in allowedUsernames:
    #    context.bot.send_message(chat_id=update.effective_chat.id, text="You are not allowed to use this bot")
    #    return
    if "Show" in update.message.text:
        context.bot.sendMediaGroup(chat_id=update.effective_chat.id,
                                   media=[InputMediaPhoto(open('out.jpg', 'rb'))])
        context.bot.sendMediaGroup(chat_id=update.effective_chat.id,
                                   media=[InputMediaPhoto(open('map.jpg', 'rb'))])


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater('2112222061:AAEU4dQirnGstVTE8jvz9MiUizAZVju3l84')

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text, messageHandler))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()