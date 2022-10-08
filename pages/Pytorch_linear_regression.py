#@TO_REMOVE
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# common imports
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

st.title("Linear Regression using Pytorch")

import torch
import torch.nn as nn
import torch.functional as F
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

# own models
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

st.write("### Training Parameters")

# define number of epochs, one epoch = passed through all training set
epochs = st.slider("Epochs", min_value=1_000, max_value=100_000, step=1_000, value=1_000)

# define learning rate
learning_rate_expo = st.slider("Learning rate exponential (10^x)", min_value=-10, max_value=-2, step=1, value=-2)
learning_rate = 10 ** learning_rate_expo

training_btn = st.button("Traing Model")

if training_btn:
    st.write("Progress")
    training_progress = st.progress(0)

    writer = SummaryWriter()

    with st.echo():
        # loss function
        loss_fn = nn.L1Loss()

        # optimizer
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=learning_rate)

        # tracking progress
        epoch_count = []
        loss_values = []
        test_loss_values = []

        for epoch in range(epochs):
            # enter training mode
            model.train()

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

            # testing
            if epoch % 10 == 0:
                with torch.inference_mode():
                    # use model to predict test data
                    test_pred = model(x_test)
                    test_loss = loss_fn(test_pred, y_test)

                    writer.add_scalar('Loss/train', loss, epoch)
                    writer.add_scalar('Loss/test', test_loss, epoch)

                    epoch_count.append(epoch)
                    loss_values.append(loss)
                    test_loss_values.append(test_loss)
                training_progress.progress(epoch/epochs)

        writer.flush()
        writer.close()

    st.write("Parameters after training")
    st.code(model.state_dict())

    initial_loss = loss_values[0].detach().item()
    final_loss = loss_values[-1].detach().item()
    st.metric('Final training loss', final_loss, final_loss - initial_loss)

    r2score = torchmetrics.R2Score()
    acc = r2score(model(x_test), y_test)
    st.metric('R2 Score', '{:.4f}%'.format(acc * 100))

    with torch.inference_mode():
        y_pred = model(x_test)
        plot_pred(y_pred=y_pred, title='Predictions after training')

    st.write("## Tracking Progress")

    plt.figure(figsize=(10, 7))
    plt.scatter(epoch_count, [x.detach().numpy() for x in loss_values], color='blue', label='Train Loss')
    plt.scatter(epoch_count, test_loss_values, color='red', label='Test Loss')
    plt.title('Train/Test Loss Progress', fontdict={'size': 22})
    plt.legend()
    st.pyplot(plt.gcf())

    st.write("## Saving model")

    MODEL_PATH = Path('cache')
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = 'pytorch_linear_regression.pt'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    torch.save(obj=model, f=MODEL_SAVE_PATH)
    st.write(f'Saved model to {MODEL_SAVE_PATH}')

    st.write("[Tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) is available, just run `tensorboard --logdir=runs` and go to http://localhost:6006")