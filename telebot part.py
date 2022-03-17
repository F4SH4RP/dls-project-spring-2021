# итак, от веба я далёк, но по созданию телеграмм ботов столько инструкций, что сделать это получилось даже у меня. 
# вот исхлдный код бота без детектора (шаг 3/4 сценарий 1)

#pip3 install PyTelegramBotAPI==2.2.3
import telebot

#!pip install numpy scipy scikit-image matplotlib pillow

#!pip install PyTelegramBotAPI
import telebot
from PIL import Image
import torch

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os


precision = 'fp32'
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
classes_to_labels = utils.get_coco_object_dictionary()
ssd_model.to('cpu')
ssd_model.eval()

API_TOKEN = 'None' # да да
bot = telebot.TeleBot(API_TOKEN)
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, """Hi UwU
    я детектирую обьекты на картинках (рисую вокруг боксы и подписываю)
    инструкция следующая
    отправляешь фото
    /detect
    /send
    бот отправляет тебе фото, на котором все обведено
    (todo - разные модели, переключаемые командой, и разные пороги фильтрации боксов)
    посмотреть классы, которые я знаю - /classes
    посмотреть инфо и сорсы - /help
    F4DiS aka F4SH4RP aka Бабенков Юрий
    stepik.org/users/303770309
    ссылка на гит (потом будет)
    """)

@bot.message_handler(commands=['classes'])
def send_welcome(message):
    bot.reply_to(message, str(classes_to_labels))

@bot.message_handler(commands=['hdd'])
def send_welcome(message):
    os.system('df -h / | grep / > hdd.txt')
    with open('hdd.txt', 'r') as file_hdd:
        data_hdd = file_hdd.read().replace('\n', '')

    bot.reply_to(message, ('Filesystem             Size  Used Avail Use% Mounted on' + "\n" + data_hdd) )

@bot.message_handler(commands=['uptime'])
def send_welcome(message):
    os.system('uptime > uptime.txt')
    with open('uptime.txt', 'r') as file_uptime:
        data_uptime = file_uptime.read().replace('\n', '')
    bot.reply_to(message, (data_uptime))

@bot.message_handler(commands=['ram'])
def send_welcome(message):
    os.system('free -h | grep Mem > mem.txt')
    with open('mem.txt', 'r') as file_mem:
        data_mem = file_mem.read().replace('\n', '')
    bot.reply_to(message, (data_mem[-7:-1] + ' free'))
    
@bot.message_handler(commands=['temperature'])
def send_welcome(message):
    os.system('sensors |grep Package > temp.txt')
    with open('temp.txt', 'r') as file_temp:
        data_temp = file_temp.read().replace('\n', '')
    bot.reply_to(message, (data_temp))

@bot.message_handler(commands=['hardware'])
def send_welcome(message):
    os.system('sensors |grep Package > temp.txt')
    os.system('free -h | grep Mem > mem.txt')
    os.system('uptime > uptime.txt')
    os.system('df -h / | grep / > hdd.txt')
    
    with open('temp.txt', 'r') as file_temp:
        data_temp = file_temp.read().replace('\n', '')
    with open('mem.txt', 'r') as file_mem:
        data_mem = file_mem.read().replace('\n', '')[-7:-1] + ' free'
    with open('uptime.txt', 'r') as file_uptime:
        data_uptime = file_uptime.read().replace('\n', '')
    with open('hdd.txt', 'r') as file_hdd:
        data_hdd = 'Filesystem             Size  Used Avail Use% Mounted on' + "\n" + file_hdd.read().replace('\n', '')
    bot.reply_to(message, (data_temp + "\n" +data_mem + "\n" +data_uptime + "\n" +data_hdd))
    

@bot.message_handler(content_types=['photo'])
def photo(message):
    print('message.photo =', message.photo)
    fileID = message.photo[-1].file_id
    print('fileID =', fileID)
    file_info = bot.get_file(fileID)
    print('file.file_path =', file_info.file_path)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('image.jpg', 'wb') as new_file:
        new_file.write(downloaded_file)
    detect(message)
    send(message)
    
    
def detect(message):
    pass # здесь будет детекция
    plt.savefig('testplot.png')
    Image.open('testplot.png').convert('RGB').save('image2.jpg','JPEG')
    #Image.open('testplot.png').save('testplot.png','PNG')


def send(message):

    chat_id = message.chat.id
    bot.send_chat_action(message.chat.id, 'upload_photo')
    
    bot.send_chat_action(chat_id, 'upload_photo')
    bot.send_chat_action(chat_id, 'upload_photo')
    img = open('image2.jpg', 'rb')
    bot.send_photo(chat_id, img)
    img.close()
    #только что мы скачали файл, который был передан нам в сообщении, в папку, из которой
    #был запущен скрипт. в кач-ве имени файла используется image.jpg, следовательно
    #каждый новый файл затирает старый, чтобы место не закончилось. 
print('ready(bot started)')
bot.polling()
