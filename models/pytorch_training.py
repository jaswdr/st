import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

class Model_training(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, input):
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)
        return output

    def _train_model(self, inputs, labels):
        optimizer = SGD(self.parameters(), lr=0.1)
        for epoch in range(100):
            total_loss = 0
            for iteration in range(len(inputs)):
                input_i = inputs[iteration]
                label_i = labels[iteration]

                output_i = self(input_i)

                loss = (output_i - label_i)**2

                loss.backward()

                total_loss += float(loss)

            if (total_loss < 0.0001):
                break

            optimizer.step()
            optimizer.zero_grad()
        return epoch
