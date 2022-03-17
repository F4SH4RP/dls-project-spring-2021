import telebot
from PIL import Image
import torch

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os


from threading import Thread
import torch, torchvision
import mmdet
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

precision = 'fp32'
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
classes_to_labels = utils.get_coco_object_dictionary()
ssd_model.to('cpu')
ssd_model.eval()

API_TOKEN = 'None' # на проде другой, естессно
bot = telebot.TeleBot(API_TOKEN)
threshold = 0.10
current_model_choise = 'ssd'
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, """Hi UwU
    я детектирую обьекты на картинках (рисую вокруг боксы и подписываю)
    инструкция следующая
    отправляешь фото
    бот отправляет тебе фото, на котором все обведено
    напиши перед фотографией rcnn чтобы обработать ею вместо ssd 
    посмотреть классы, которые я знаю - /classes
    посмотреть инфо и сорсы - /help
    F4DiS aka F4SH4RP aka Бабенков Юрий
    stepik.org/users/303770309
    https://github.com/F4SH4RP/Detection_for_dls
    """)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, """hardware spectrating:
/ram /hdd/ temperature /uptime
/hardware
list all classes i can detect - /classes
switch models - /ssd /rcnn
WARNING /rcnn DO NOT WORK PROPERLY FOR NOW, хотя попробовать разрешаю
/threshold x to filter boxes
    """)

@bot.message_handler(commands=['classes'])
def send_welcome(message):
    bot.reply_to(message, str(classes_to_labels)) # преобразует массив в строку и отправит. можно красивее (без скобок) но нужно ли?
    
# немножко простых команд для отслеживания состояния железа
@bot.message_handler(commands=['hdd'])
def send_welcome(message):
    os.system('df -h / | grep / > hdd.txt')
    with open('hdd.txt', 'r') as file_hdd:
        data_hdd = file_hdd.read().replace('\n', '')
    bot.reply_to(message, ('Filesystem             Size  Used Avail Use% Mounted on' + "\n" + data_hdd) )

@bot.message_handler(commands=['ram'])
def send_welcome(message):
    os.system('free -h | grep Mem > mem.txt')
    with open('mem.txt', 'r') as file_mem:
        data_mem = file_mem.read().replace('\n', '')
    bot.reply_to(message, (data_mem[-7:-1] + ' free'))

@bot.message_handler(commands=['uptime'])
def send_welcome(message):
    os.system('uptime > uptime.txt')
    with open('uptime.txt', 'r') as file_uptime:
        data_uptime = file_uptime.read().replace('\n', '')
    bot.reply_to(message, (data_uptime))
    
@bot.message_handler(commands=['temperature'])
def send_welcome(message):
    os.system('sensors |grep Package > temp.txt')
    with open('temp.txt', 'r') as file_temp:
        data_temp = file_temp.read().replace('\n', '')
    bot.reply_to(message, (data_temp))
# и теперь все вместе
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
    

    
# фича, меняющая глобально уровень фильтрации боксов.
@bot.message_handler(commands=['threshold'])
def send_welcome(message):
    global threshold
    try:
        if 0 < int(str(message.text)[11:]) < 99:
            bot.reply_to(message, 'treshhold now is ' + (str(message.text)[11:]))
            threshold = int(str(message.text)[11:])/100
        else:
            bot.reply_to(message, 'dude, it should be from 1 to 99')
    except:
        bot.reply_to(message, 'usage: /threshold x, 0<x<99')

# фича, глобально меняющая используемые модели
@bot.message_handler(commands=['rcnn'])
def send_welcome(message):
    global current_model_choise
    current_model_choise = 'rcnn'


@bot.message_handler(commands=['ssd'])
def send_welcome(message):
    global current_model_choise
    current_model_choise = 'ssd'


# получает фото, обрабатывает одной из моделей и отсылает
@bot.message_handler(content_types=['photo'])
def photo(message):
    global current_model_choise
    print('message.photo =', message.photo)
    fileID = message.photo[-1].file_id
    print('fileID =', fileID)
    file_info = bot.get_file(fileID)
    print('file.file_path =', file_info.file_path)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('image.jpg', 'wb') as new_file:
        new_file.write(downloaded_file)
    if current_model_choise == 'rcnn':
        detect2(message)
    else:
        detect(message)
    send(message)
    
# модель ссд
def detect(message):
    global threshold
    uris = ['image.jpg']
    inputs = [utils.prepare_input(uri) for uri in uris]
    tensor = utils.prepare_tensor(inputs, precision == 'fp16')
    with torch.no_grad():
        detections_batch = ssd_model(tensor)
    results_per_input = utils.decode_results(detections_batch)
    best_results_per_input = [utils.pick_best(results, threshold) for results in results_per_input]
    for image_idx in range(len(best_results_per_input)):
        fig, ax = plt.subplots(1)
        # Show original, denormalized image...
        image = inputs[image_idx] / 2 + 0.5
        ax.imshow(image)
        # ...with detections
        bboxes, classes, confidences = best_results_per_input[image_idx]
        for idx in range(len(bboxes)):
            left, bot, right, top = bboxes[idx]
            x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
    #plt.show()
    plt.savefig('testplot.png')
    Image.open('testplot.png').convert('RGB').save('image2.jpg','JPEG')
    #Image.open('testplot.png').save('testplot.png','PNG')
    

# модель rcnn
def detect2(message):
    global threshold
    config = 'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
    checkpoint = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
    model = init_detector(config, checkpoint, device='cpu')
    img = 'image.jpg'
    result = inference_detector(model, img)

#    os.system('spectacle -a -d 2 -o /home/a/image2.jpg') OMFG thats unreal hack
# вообще я до сих пор удивляюсь, как такое могло прийти мне в голову: открывать окно матплотлиба и автоматом делать его скрин
# и уже его отсылать. самое ужасное, у меня почти получилось. хорошо, что почти
# замените в mmdet/core/visualisation/image.py строку plt.show() на сохранение в картинку по подобию detect() выше
    show_result_pyplot(model, img, result, score_thr=threshold)
#    img2 = show_result_pyplot(model, img, result, score_thr=0.3)
    
# функция отправки сообщения
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
