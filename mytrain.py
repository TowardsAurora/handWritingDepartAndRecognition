import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image

from handWritingDepartAndRecognition.model.cnn import cnn


def count_types():
    count = 0
    for _ in os.listdir('./data/train'):
        count += 1
    return count


count_types_of_train = count_types()

data_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder('./data/train', transform=data_transform)
test_dataset = datasets.ImageFolder('./data/train', transform=data_transform)
predict_dataset = datasets.ImageFolder('./data/predict', transform=data_transform)
# predict_dataset = datasets.ImageFolder('./data/predictshf', transform=data_transform)  # 白小松和曹佳睿图片位置进行更换

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=53, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=53, shuffle=True)
predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=1, shuffle=True)
classes = train_dataset.classes


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--epochs', type=int, default=20, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='')

    opt = parser.parse_args()

    return opt


def read_img(path):
    image = Image.open(path)

    # 对图像进行预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale()
    ])
    image = transform(image)

    # 增加批量维度
    image = image.unsqueeze(0)
    return image


def pred(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    image = image.to(device)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    # print('Predicted class:',predicted, predicted[0])
    i = int(str(predicted[0]).split('(')[1].split(',')[0])
    print('predicted class:', classes[i])


def pred_index(index):
    load_class = classes[index]
    print('loaded class:', load_class)
    path = './data/train/' + load_class + '/一.png'

    image = read_img(path)

    predicted_gpu_part_single(image)


def train_gpu_part():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = parse_option()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=opt.learning_rate,
                                 betas=(0.9, 0.999)
                                 )

    for epoch in range(opt.epochs):
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device), labels.to(device)
            predicted_labels = model(images)
            loss = criterion(predicted_labels, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), "saved_model/model.pth")
        test_acc, test_loss = test_gpu_part()
        theta = 0.985
        if test_acc > theta:
            break
        time.sleep(1)
        print('epoch:{}  train_loss:{:.4f}'.format(epoch + 1, loss))
        print('test_acc:{:.2f}%  test_loss:{:.4f}'.format(test_acc * 100, test_loss))
        time.sleep(1)
    torch.save(model.state_dict(), "saved_model/model.pth")
    print("最终模型存储完成")


def test_gpu_part():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    model.eval()
    model.load_state_dict(torch.load('./saved_model/model.pth'))
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    for i, (images, labels) in enumerate(tqdm(test_loader)):
        images, labels = images.to(device), labels.to(device)
        predicted_labels = model(images)
        loss = criterion(predicted_labels, labels)
        _, predicted = torch.max(predicted_labels.data, 1)
        total += labels.size(0)
        # print('pred:{}  labels:{}'.format(predicted, labels))
        correct += (predicted == labels).sum().item()
    acc = correct / total
    print(correct, total)
    print('accuracy:{:.2f}%   loss:{:.4f}'.format(acc * 100, loss))
    return acc, loss


def test_gpu_part_depart_classes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    model.eval()
    model.load_state_dict(torch.load('./saved_model/model.pth'))
    # criterion = nn.CrossEntropyLoss()

    class_correct = list(0. for i in range(count_types_of_train))
    class_total = list(0. for i in range(count_types_of_train))

    for i, (images, labels) in enumerate(tqdm(test_loader)):
        images, labels = images.to(device), labels.to(device)
        predicted_labels = model(images)
        # loss = criterion(predicted_labels, labels)
        _, predicted = torch.max(predicted_labels.data, 1)

        c = (predicted == labels).squeeze()

        for j in range(len(labels)):
            label = labels[j]
            class_correct[label] += c[j].item()
            class_total[label] += 1
    for i in range(count_types_of_train):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


def predicted_gpu_part_single(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    model.eval()
    model.load_state_dict(torch.load('./saved_model/model.pth'))

    image = image.to(device)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    # print('Predicted class:',predicted, predicted[0])
    i = int(str(predicted[0]).split('(')[1].split(',')[0])
    print('predicted class:', classes[i])


def predicted_gpu_part_multi():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    model.eval()
    model.load_state_dict(torch.load('./saved_model/model.pth'))

    for i, (images, labels) in enumerate(predict_loader):
        images, labels = images.to(device), labels.to(device)
        predicted_labels = model(images)
        _, predicted = torch.max(predicted_labels.data, 1)
        i = int(str(labels[0]).split('(')[1].split(',')[0])
        j = int(str(predicted[0]).split('(')[1].split(',')[0])
        bool_tensor = predicted == labels
        conBool = str(bool_tensor).split('[')[1].split(']')[0]
        print('loaded class:{}   predicted class:{}   conformance:{}'.format(classes[i], classes[j], conBool))


if __name__ == '__main__':
    print("开始训练模型")
    train_gpu_part()
    #
    print("开始测试模型准确率")
    test_gpu_part()
    #
    print("开始预测多图片所属类别")
    predicted_gpu_part_multi()
    #
    print("开始预测单图片所属类别")
    pred_index(0)
    print("开始测试每个类别分类的准确率")
    test_gpu_part_depart_classes()
