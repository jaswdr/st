#@TO_REMOVE
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# common imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

st.title("Linear Regression using Pytorch")

import torch
import torch.nn as nn
import torch.functional as F
from models.pytorch_linear_regression import LinearRegressionModel

torch.manual_seed(13)
model = LinearRegressionModel()

st.write("## Model")

with open('models/pytorch_linear_regression.py', 'r') as f:
    st.code(f.read())

st.write("Model parameters")
st.code(model.state_dict())

st.write("## Training")

# know best parameters
weight = torch.tensor(.7)
bias = torch.tensor(.3)

x = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y = (weight * x) + bias

x_train, x_test = torch.split_with_sizes(x, (40, 10))
y_train, y_test = torch.split_with_sizes(y, (40, 10))

def plot_pred(
    x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
    y_pred=None, title=''):
    plt.figure(figsize=(10, 7))
    plt.scatter(x_train, y_train, color='blue', label='Train')
    plt.scatter(x_test, y_test, color='red', label='Test')
    if y_pred != None:
        plt.scatter(y_test, y_pred.squeeze().tolist(), color='green', label='Predictions')
    plt.title(title, fontdict={'size': 22})
    plt.legend()
    st.pyplot(plt.gcf())

with torch.inference_mode():
    y_pred = model(x_test)
    plot_pred(y_pred=y_pred, title='Predictions before training')

with st.spinner("Trainning model..."):
    with st.echo():
        # loss function
        loss_fn = nn.L1Loss()

        # optimizer
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=0.01)

        # define number of epochs, one epoch = passed through all training set
        epochs = 1000

        # enter training mode
        model.train()

        for epoch in range(epochs):
            # get results
            y_pred = model(x_train)
            
            # find loss
            loss = loss_fn(y_pred, y_train)
            
            # cleanup optimizer
            optimizer.zero_grad()

            # apply back propagation
            loss.backward()

            # apply parameters optimization
            optimizer.step()

        # go back to inference mode
        model.eval()

with torch.inference_mode():
    plot_pred(y_pred=model(y_test), title='Predictions after training')