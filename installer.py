import os
print('проверка системных требований')
if (os.system('uname -o') == 0):
    print('OS : ok')
os.system('pip3 install torch torchvision numpy scipy matplotlib pillow tk')
import torch

print('Python libs : ok')
print('скачиваем модель')
precision = 'fp32'
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
classes_to_labels = utils.get_coco_object_dictionary()

print('модифицируем модель')

os.system('mv ~/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Detection/SSD/src/entrypoints.py ~/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Detection/SSD/src/entrypoints1.py')

os.system('cp  entrypoints.py ~/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/Detection/SSD/src/entrypoints.py')

print('подготовка завершена. теперь можно запускать приложение')
