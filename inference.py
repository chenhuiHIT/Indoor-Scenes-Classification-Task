# from __future__ import print_function, division

import torch
import torch.nn as nn

from torchvision import models, transforms

from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Predict given image')
parser.add_argument('--path',  type=str,
                    help='image file path')

model_ft = models.resnet50()
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 67)
model_ft.load_state_dict(torch.load('./model/bst_resnet50.pt'))
model_ft.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

categories = ['airport_inside', 'artstudio', 'auditorium', 'bakery', 'bar', 'bathroom', 'bedroom', 'bookstore',
              'bowling', 'buffet', 'casino', 'children_room', 'church_inside', 'classroom', 'cloister', 'closet',
              'clothingstore', 'computerroom', 'concert_hall', 'corridor', 'deli', 'dentaloffice', 'dining_room',
              'elevator', 'fastfood_restaurant', 'florist', 'gameroom', 'garage', 'greenhouse', 'grocerystore',
              'gym', 'hairsalon', 'hospitalroom', 'inside_bus', 'inside_subway', 'jewelleryshop', 'kindergarden',
              'kitchen', 'laboratorywet', 'laundromat', 'library', 'livingroom', 'lobby', 'locker_room', 'mall',
              'meeting_room', 'movietheater', 'museum', 'nursery', 'office', 'operating_room', 'pantry', 'poolinside',
              'prisoncell', 'restaurant', 'restaurant_kitchen', 'shoeshop', 'stairscase', 'studiomusic', 'subway',
              'toystore', 'trainstation', 'tv_studio', 'videostore', 'waitingroom', 'warehouse', 'winecellar']


if __name__ == '__main__':
    args = parser.parse_args()
    filename = args.path
    try:
        input_image = Image.open(filename)
    except:
        raise IOError('can not load image file')

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model_ft.to('cuda')

    with torch.no_grad():
        output = model_ft(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    print(categories[top1_catid[0]], top1_prob[0].item())



