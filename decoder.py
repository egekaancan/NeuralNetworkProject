# Libraries are imported
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# RNN class uses LSTM and Word Embedding to predict the captions of the images
class RNN(torch.nn.Module):

    # i_size is input size(Since we are using 300 dimemsional embedding matrix, it will be 300 as well)
    # h_size is number of hidden units in LSTM cells
    # o_size is output size which is 1004 in our case
    def __init__(self, i_size, h_size, o_size, layer_num=2):
        super(RNN, self).__init__()
        self.layer_num = layer_num
        self.h_size = h_size
        self.i_size = i_size
        self.o_size = o_size
        # We are using LSTMCell instead of LSTM since we need to feed LSTMs with output of previous LSTM in prediction part
        self.lstm = torch.nn.LSTMCell(input_size=self.i_size, hidden_size=self.h_size)
        # Last added fully connected layer to LSTM cells
        self.linear = torch.nn.Linear(self.h_size, self.o_size)
        # Embedding layer of torch is initialized
        self.embedding = torch.nn.Embedding(num_embeddings=1004, embedding_dim=self.i_size)
        # Pretrained embedding matrix from glove is loaded
        glove = np.load('embedding_matrix.npy')
        self.embedding.weight.data.copy_(torch.from_numpy(glove))
        # Learning progress of embedding matrix is closed
        self.embedding.weight.requires_grad = False

    # Forward propagation of the decoder part
    def forward(self, x, y):
        # Hidden and Cell states are initialized
        h_state = torch.zeros(x.size(0), self.h_size).to(device)
        c_state = torch.zeros(x.size(0), self.h_size).to(device)
        # Embeddings are extracted according to words in the caption
        word_vecs = self.embedding(y)
        # output array is initialized
        o_arr = torch.empty((100, 17, 1004)).to(device)
        # First LSTM cell takes input from encoded image, other cells take input from embedding matrix
        for i in range(17):
            if i == 0:
                h_state, c_state = self.lstm(x, (h_state, c_state))
            else:
                h_state, c_state = self.lstm(word_vecs[:, i-1, :], (h_state, c_state))
            # Output of the LSTM cells are input of the fully connected layer
            o_linear = self.linear(h_state)
            # Output of the fully connected layer is the captions that is predicted in training process
            o_arr[:, i, :] = o_linear
        return o_arr

    # this function predicts the caption when encoded image is given into
    def predict(self, x):
        # Hidden and Cell states are initialized
        h_state = torch.zeros(x.size(0), self.h_size).to(device)
        c_state = torch.zeros(x.size(0), self.h_size).to(device)
        # output array is initialized
        o_arr = torch.empty((1, 17, 1004)).to(device)
        # First LSTM cell takes input from encoded image
        # However other cells are taken input from embedded vector of output of previous LSTM cell
        # Rest of the function is the same as forward propagation part
        for i in range(18):
            if i == 0:
                h_state, c_state = self.lstm(x, (h_state, c_state))
                o_linear = self.linear(h_state)
            else:
                h_state, c_state = self.lstm(self.embedding(torch.argmax(o_linear, dim=1)), (h_state, c_state))
                o_linear = self.linear(h_state)
                o_arr[:, i - 1, :] = o_linear

        return o_arr
