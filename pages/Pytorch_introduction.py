#@TO_REMOVE
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# common imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

st.title("StatQuest - Pytorch Introduction")

st.video('https://www.youtube.com/watch?v=FHdlXe1bSe4')

import torch
from models.pytorch import Model

@st.experimental_singleton
def get_model():
    return Model()

st.write('## Final Model')
with open('models/pytorch.py', 'r') as f:
    st.code(f.read())

st.write('## Inputs')
input_doses = torch.linspace(start=0, end=1, steps=11)
st.code(input_doses)

st.write('## Output')
output_values = get_model()(input_doses)
st.line_chart(output_values)

st.write('## Testing')
value = st.slider(label='Input', min_value=0., max_value=1., step=.05)
st.write('Predicted Value:')
pred = get_model()(value)
st.code(pred)

st.write('## Training')

from models.pytorch_training import Model_training
model_train = Model_training()

st.write('### Model before training')
output_values = model_train(input_doses)
st.line_chart(output_values.detach())

btn_train = st.button("Train Model")

if btn_train:
    with st.spinner('Trainning model, please wait...'):
        inputs = torch.tensor([0., .5, 1.])
        labels = torch.tensor([0., 1., 0.])
        total_epochs = model_train._train_model(inputs, labels)
        st.write(f'Done in {total_epochs} epochs.')

    st.write('### Model after training')
    output_values = model_train(input_doses)
    st.line_chart(output_values.detach())
