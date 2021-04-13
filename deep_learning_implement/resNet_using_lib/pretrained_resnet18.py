# -*- coding: utf-8 -*-
"""pretrained_resNet18.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1u8B6KnlRAEDm8Yu3zuohwWm8zPkyTKeB
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

from urllib.request import urlretrieve
import json

imagenet_json, _ = urlretrieve('http://www.anishathalye.com/media/2017/07/25/imagenet.json')
with open(imagenet_json) as f:
  imagenet_labels = json.load(f)

print(imagenet_labels[18])

preprocess = transforms.Compose([
                                 transforms.Resize(256), #이미지 크기 변경
                                 transforms.CenterCrop(224), #이미지의 중앙 부분을 잘라서 크기 조절
                                 transforms.ToTensor(), #torch.Tensor 형식으로 변경 [0,255] -> [0,1]
])

import matplotlib.pyplot as plt
import PIL

def image_loader(path):
  image = PIL.Image.open(path)
  image = preprocess(image).unsqueeze(0)
  return image.to(device,torch.float)

url = 'data:image/jpeg;base64,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'
image_path, _ = urlretrieve(url)
image = image_loader(image_path)

def imshow(tensor):
  image = tensor.cpu().clone() #matplotlib는 cpu기반
  image = image.squeeze(0)
  image = transforms.ToPILImage()(image)
  plt.imshow(image)

plt.figure()
imshow(image)

class Normalize(nn.Module):
  def __init__(self,mean,std):
    super(Normalize,self).__init__()
    self.register_buffer('mean',torch.Tensor(mean))
    self.register_buffer('std',torch.Tensor(std))

  def forward(self,input):
    mean = self.mean.reshape(1,3,1,1)
    std = self.std.reshape(1,3,1,1)
    return (input-mean)/std

model = nn.Sequential(
    Normalize(mean = [0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    torch.hub.load('pytorch/vision:v0.6.0','resnet18',pretrained=True)
).to(device).eval()

outputs = model(image)
percentages = torch.nn.functional.softmax(outputs,dim=1)[0] * 100

for i in outputs[0].topk(5)[1]:
    # 높은 값을 가지는 순서대로 인덱스에 해당하는 클래스 이름과, 그 확률 값 출력하기
    print(f"인덱스: {i.item()} / 클래스명: {imagenet_labels[i]} / 확률: {round(percentages[i].item(), 4)}%")

