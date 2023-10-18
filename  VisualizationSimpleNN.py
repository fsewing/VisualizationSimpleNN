import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import tkinter as tk
from tkinter import ttk, Canvas, PhotoImage, mainloop
import random
import math
from PIL import Image, ImageDraw, ImageTk

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('cat.txt', delimiter=',')
x = dataset[:,0:2]
y = dataset[:,2]

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

print(len(x))

class Fourier(nn.Module):
    
    def __init__(self, nmb=5, scale=10):
        super(Fourier, self).__init__()
        self.b = torch.randn(2 , nmb) * scale
        self.pi = 3.14159265359
   
    def forward(self, v):
        x_proj = torch.matmul(2 * self.pi * v, self.b)
        return torch.cat(([torch.sin(x_proj), torch.cos(x_proj)]), -1)

# classifier class

class Classifier(nn.Module):
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()
        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        # create loss function
        self.loss_function = nn.MSELoss()
        # create optimiser, using simple stochastic gradient descent
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        # counter and accumulator for progress
        self.counter = 0
        self.progress = []
        pass  
    def forward(self, inputs):
        # simply run model
        return self.model(inputs)
    def train(self, inputs, targets):
        # calculate the output of the network
        outputs = self.forward(inputs)
        # calculate loss
        loss = self.loss_function(outputs, targets)
        # increase counter and accumulate error every 10
        # self.counter += 1
        # if (self.counter % 10 == 0):
        #     self.progress.append(loss.item())
        #     pass
        # if (self.counter % 10000 == 0):
        #     print("counter = ", self.counter)
        #     pass
        # zero gradients, perform a backward pass, and update the weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pass
    def plot_progress(self):
        pass
    pass


# Watch the NN learn

WIDTH, HEIGHT = 400, 400
img_w, img_h = 28,28

root = tk.Tk()
root.geometry("400x500")


pilImage = Image.new("RGB", (img_w, img_h), "#cccccc")  # create new Image
pilImageResize = Image.new("RGB", (WIDTH, HEIGHT), "#cccccc")  # create new Image

pilImageResize = pilImage.resize((WIDTH, HEIGHT), Image.Resampling.BOX)
tkImage = ImageTk.PhotoImage(pilImageResize)

canvas = Canvas(root, width=WIDTH, height=HEIGHT, bg="#f00000")
canvas.pack()
image_container = canvas.create_image((WIDTH/2, HEIGHT/2), image=tkImage, state="normal")


# create FFT

F = Fourier()

# create neural network

C = Classifier()

def start():
    # train network
    epochs = 1000

    for i in range(epochs):
        dctx = ImageDraw.Draw(pilImage)
        print('training epoch', i+1, "of", epochs)
        for j in range(0, len(x)):
            rnd = random.randint(0, len(x)-1)
            C.train(F.forward(x[rnd]), y[rnd])
            new_color = C.forward(x[rnd]).float()*255
            new_color = int(new_color)
            red, green, blue = new_color ,new_color, new_color
            dctx.point((x[rnd][0].int().item(),x[rnd][1].int().item()), fill="#%02x%02x%02x" % (red, green, blue))
        del dctx
        pilImageResize = pilImage.resize((400, 400), Image.Resampling.BOX)
        tkImage = ImageTk.PhotoImage(pilImageResize)
        canvas.itemconfig(image_container,image=tkImage)
        root.update()


ttk.Button(root, text= "Start", command= start).pack()
root.mainloop()