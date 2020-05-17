from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import telegram
TOKEN = '1193023091:AAEl9eLOZ6Q0PdDRXF07TprHDXt9tEGuclo'
chat_id = 1213182814
bot = telegram.Bot(TOKEN)
print(bot.get_me())
{'id': 514487748, 'first_name': 'trial1', 'is_bot': True, 'username': 'trial3700_bot'}
link='https://bitcoin.org/img/icons/opengraph.png'
bot = telegram.Bot(TOKEN)

def tasveer(name,caption):
  bot.send_photo(chat_id=chat_id, photo=open(name, 'rb'))
  bot.send_message(chat_id=chat_id, text="Alert! Motion detected in Room! "+caption)

def start(bot, update):
  update.message.reply_text("I'm a bot, Nice to meet you!obey")

def status(bot,update):
  update.message.reply_text("Room status: Unoccupied")
  
def convert_uppercase(bot, update):
  update.message.reply_text(update.message.text.upper())

def send_image(bot,update,link):
  bot.send_photo(link)

def main():
  # Create Updater object and attach dispatcher to it
  updater = Updater(TOKEN)
  dispatcher = updater.dispatcher
  print("Bot started")

  # Add command handler to dispatcher
  start_handler = CommandHandler('start',start)
  upper_case = MessageHandler(Filters.text, convert_uppercase)
  dispatcher.add_handler(start_handler)

  # Adding status command handler
  status_handler = CommandHandler('status',status)
  dispatcher.add_handler(status_handler)
  dispatcher.add_handler(upper_case)
 
  #pic = 'https://bitcoin.org/img/icons/opengraph.png'
  #bot.send_photo(chat_id, pic)
  #Adding the functionality of sending images--------------------
#   bot = telegram.Bot(TOKEN)
#   if bot.get_updates():
#       chat_id = bot.get_updates()[-1].message.chat_id
#       # Enter text
#       txt = input("Enter your Bot's reply text: \n")
# # send message
#       bot.send_message(chat_id, txt)
# #Get pic link from Google.'
#       pic = 'https://bitcoin.org/img/icons/opengraph.png'
#       bot.send_photo(chat_id, pic)
#   else:
#     print("Empty list. Please, chat with the bot")

  # Start the bot
  updater.start_polling()

if __name__ == '__main__':
  main()
