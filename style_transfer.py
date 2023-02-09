import torch
import torch as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T

from imageio import imread

import numpy as np
import matplotlib.pyplot as plt

"""Style transfer is an technique that uses two images (content, style) to create a third artistic one. the is done using the feature maps of a pretrained convolutional neural network. this helps to stablish similierities between both images.

The paper on style transfer was published in 2015 by Gaty el al, and in this publication VGG19 pretrained model was used, here we will do the same.

"""

vgg_model = torchvision.models.vgg19(pretrained=True)

for p in vgg_model.parameters():
  p.requires_grad = False

vgg_model.eval()

#"http://www.nigerianembassy.co.il/wp-content/uploads/2019/04/flag.jpg"

content = imread("https://media.istockphoto.com/id/1077585330/photo/oranges.jpg?s=612x612&w=0&k=20&c=ZfBi_hD6A_tvNoanVwYBVZIgtcUxZEMzi11DUsGAK4M=")
style_design = imread("https://www.shutterstock.com/image-vector/african-print-fabric-vector-seamless-600w-2022609896.jpg")
target_canvas = np.random.randint(0, 255, size=content.shape, dtype=np.uint8)

transform = T.Compose([T.ToTensor(),
                       T.Resize(256),
                       T.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
                       ])


content =transform(content).unsqueeze(0)
style_design =transform(style_design).unsqueeze(0)
target_canvas =transform(target_canvas).unsqueeze(0)

print(target_canvas.shape)
print(style_design.shape)
print(content.shape)

def Visualize_img(target_canvas, style_design, content):

  """
  Visualize input picture, style and canvas
  """

  fig, ax = plt.subplots(1,3, figsize=(18,6))

  pic = content.squeeze().numpy().transpose(1,2,0)
  pic= (pic-np.min(pic)) / (np.max(pic) - np.min(pic))

  ax[0].imshow(pic)
  ax[0].set_title("Content Image")

  pic = style_design.squeeze().numpy().transpose(1,2,0)
  pic= (pic-np.min(pic)) / (np.max(pic) - np.min(pic))

  ax[1].imshow(pic)
  ax[1].set_title("style Design")

  pic = target_canvas.squeeze().numpy().transpose(1,2,0)
  pic= (pic-np.min(pic)) / (np.max(pic) - np.min(pic))

  ax[2].imshow(pic)
  ax[2].set_title("target Canvas")

  plt.show()

def featureMap(image, model):

  feature_name = []
  feature_maps = []

  layer_number = 0

  """
  Loop through network layers and select only 
  convolutional blocks

  """

  for layer in range(len(model.features)):
    image = model.features[layer](image)

    if "Conv2d" in str(model.features[layer]):
      feature_maps.append(image)

      feature_name.append("conv_"+ str(layer_number))
      layer_number +=1

  return feature_maps, feature_name




def gram_matrix(map):


  """
  Applying Gram matrix to the convolutional features extracted
  helps to make style data needed for transfer
  """

  _, channels, height, width = map.shape
  map = map.reshape(channels, height*width)

  # get convariance matrix
  gram = torch.mm(map,map.t())/ (channels*height*width)
  
  return gram



content_layer = ["conv_1", "conv_2"]

style_layer = ["conv_1", "conv_3", "conv_5","conv_7", "conv_8" ]

style_weights = [1,  0.6,   0.5,      0.4,     0.2,      1]

content_feature_maps, content_feature_name = featureMap(content, vgg_model)
style_feature_maps, style_feature_name = featureMap(style_design, vgg_model)

style_scale = 1e7 #Experimental number, larger number results to stronger style display

target_image = target_canvas.clone()
target_image.requires_grad = True

num_epochs = 1200

optimizer = torch.optim.RMSprop([target_image], lr=0.005)



for e in range(num_epochs):

  content_loss = 0
  style_loss = 0

  target_feature_maps, target_feature_name = featureMap(target_image, vgg_model)

  for layer in range(len(target_feature_name)):

    if target_feature_name[layer] == content_layer:
      content_loss+= torch.mean((target_feature_maps[layer]
                                 -content_feature_maps[layer])**2) #MSE
    
    
    if target_feature_name[layer] in style_layer:
      

      gram_target = gram_matrix(target_feature_maps[layer])
      gram_style = gram_matrix(style_feature_maps[layer])

      style_loss += torch.mean( (gram_target-gram_style)**2) * style_weights[style_layer.index(target_feature_name[layer])]

  combined_loss = style_scale*style_loss + content_loss

  optimizer.zero_grad()
  combined_loss.backward()
  optimizer.step()



def result_visualization(content, target, style ):

  """
  Visualize the result after styling.
  """


  fig,ax = plt.subplots(1,3,figsize=(18,11))

  pic = content.squeeze().numpy().transpose((1,2,0))
  pic = (pic-np.min(pic)) / (np.max(pic)-np.min(pic))
  ax[0].imshow(pic)
  ax[0].set_title('Content',fontweight='bold')


  pic = torch.sigmoid(target).detach().squeeze().numpy().transpose((1,2,0))
  ax[1].imshow(pic)
  ax[1].set_title('Target',fontweight='bold')


  pic = style.squeeze().numpy().transpose((1,2,0))
  pic = (pic-np.min(pic)) / (np.max(pic)-np.min(pic))
  ax[2].imshow(pic)
  ax[2].set_title('Style',fontweight='bold')


  plt.show()

result_visualization(content, target_image, style_design )
