# GAN_Masked_Face

## Introduction
Facial recognition has been used everywhere today. However, during the COVID pandemic, as people are wearing masks everywhere, it is hard for facial recognition algorithms to achieve a high accuracy. As researchers are implementing and developing new algorithms for masked faces, usually naive simulated masked face images are being used for training. These images do not look real. A generative adversarial network (GAN) is a class of machine learning frameworks, where two neural networks contest with each other in a game. GAN has been widely used to produce realistic sythesized images, for example, CycleGAN for transferring image styles, StarGAN for changing facial expression and attributes. Therefore, we would like to develop a GANbased masked face image generator to make a better simulation and other researchers would benefit from better quality data.

## Install Requirements
The code is tested with Python 3.7.5 and the packages listed in requirements.txt. It is advisiable to make a new virtual environment and install the dependencies. 

The provided requirements.txt can be used to install all the required packages. Use the following commands to install dependencies:

for linux:

'''
pip install –r requirements.txt
'''
for windows:

'''
py -3 -m pip install -r requirements.txt
'''

## Run the Program

Run train.py as the top module