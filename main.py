# Library Import
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset import ImageData
from torch.utils.data import DataLoader
import pandas as pd
import warnings
import os
from decoder import RNN
from resnet import CNN
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Multithread is used in dataloader part
    torch.multiprocessing.freeze_support()
    # Cuda device is used (We used NVidia RTX2060 as device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset Import
    train = h5py.File('eee443_project_dataset_train.h5', 'r')
    test = h5py.File('eee443_project_dataset_test.h5', 'r')

    # Dataset is seperated(train_ims and test_ims has never been used)
    train_cap = np.array(train['train_cap'])
    train_imid = np.array(train['train_imid'])
    train_url = np.array(train['train_url'])

    test_cap = np.array(test['test_caps'])
    test_imid = np.array(test['test_imid'])
    test_url = np.array(test['test_url'])

    # Train and Validation sets are seperated randomly with rates of 85% training and 15% validation
    x_train, x_val, y_train, y_val = train_test_split(train_imid, train_cap, test_size=0.15, random_state=10)

    # Datasets for all parts take images from jpg file that are previously downloaded
    # Dataloader helps us to create batches and work on GPU(Multithreading)
    val_data = ImageData(directory=os.getcwd() + "/train_images", imid=x_val, cap=y_val)
    val_loader = DataLoader(dataset=val_data, batch_size=1, num_workers=4)

    train_data = ImageData(directory=os.getcwd() + "/train_images", imid=x_train[:340100], cap=y_train[:340100])
    train_loader = DataLoader(dataset=train_data, batch_size=100, num_workers=4)

    test_data = ImageData(directory=os.getcwd() + "/test_images", imid=test_imid, cap=test_cap)
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=4)

    # Word codes are connected to the words
    warnings.filterwarnings("ignore")
    words = pd.read_hdf("eee443_project_dataset_train.h5", 'word_code')
    cols = list(words.columns)
    word_nums = words.values
    dictionary = {}
    for i, word in enumerate(cols):
        dictionary[word_nums[0][i]] = word

    # Encoder part which uses Resnet152 is initialized
    cnn = CNN().to(device=device)
    # Decoder part which uses LSTM and Word Embedding Matrix obtained from Glove is initialized
    rnn = RNN(i_size=300, h_size=256, o_size=1004).to(device)
    # Cross Entropy Loss is chosen as the loss function
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # Adan optimizer is chosen and learning rate for the training part is 7e-3
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, rnn.parameters()), lr=7e-3)

    print("Training Part Started")
    # 3 Epoch is used in training part
    for j in range(3):
        print("Epoch:", j+1)

        for i, samples in enumerate(train_loader):

            rnn.zero_grad()
            rnn.train()
            cnn.train()

            # Forward Propagation
            sample_ims = samples[0].float().cuda()
            sample_caps = samples[1].to(torch.int64).cuda()
            encoded = cnn(sample_ims)
            decoded = rnn(encoded, sample_caps.to(device))

            # Loss Calculation
            loss = criterion(decoded.view(-1, 1004), sample_caps.long().to(device).contiguous().view(-1))
            # Backward Propagation
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print("Epoch:", j+1)
                print("Iteration:", i+1)
                print("Cross Entropy Loss:", loss.item())

    # Trained Networks are saved
    torch.save(rnn.state_dict(), "rnn.pth")
    torch.save(cnn.state_dict(), "cnn.pth")

    # Trained networks are loaded
    cnn.load_state_dict(torch.load("cnn.pth"))
    rnn.load_state_dict(torch.load("rnn.pth"))

    # Average validation loss of the trained model
    val_loss = 0
    print("Validation Loss Calculation Started")
    for i, samples in enumerate(val_loader):
        if i % 1000 == 0:
            print("Validation Iteration", i)
        rnn.eval()
        cnn.eval()
        # Forward Propagation
        sample_ims = samples[0].float().cuda()
        sample_caps = samples[1].to(torch.int64).cuda()
        encoded = cnn(sample_ims)
        decoded = rnn.predict(encoded)
        # Loss Calculation
        valLoss = criterion(decoded.view(-1, 1004), sample_caps.long().to(device).contiguous().view(-1))
        val_loss += valLoss.item() / y_val.shape[0]
    print("Validation Loss:", val_loss)

    # Average test loss of trained model
    test_loss = 0
    print("Test Loss Calculation Started")
    for i, samples in enumerate(test_loader):
        if i % 1000 == 0:
            print("Test Iteration", i)
        rnn.eval()
        cnn.eval()
        # Forward Propagation
        sample_ims = samples[0].float().cuda()
        sample_caps = samples[1].to(torch.int64).cuda()
        encoded = cnn(sample_ims)
        decoded = rnn.predict(encoded)
        # Loss Calculation
        testLoss = criterion(decoded.view(-1, 1004), sample_caps.long().to(device).contiguous().view(-1))
        test_loss += testLoss.item() / test_cap.shape[0]
    print("Test Loss:", test_loss)

    val_data = ImageData(directory=os.getcwd() + "/train_images", imid=x_val, cap=y_val)
    val_loader = DataLoader(dataset=val_data, batch_size=1, num_workers=4, shuffle=True)

    train_data = ImageData(directory=os.getcwd() + "/train_images", imid=x_train[:340100], cap=y_train[:340100])
    train_loader = DataLoader(dataset=train_data, batch_size=1, num_workers=4, shuffle=True)

    test_data = ImageData(directory=os.getcwd() + "/test_images", imid=test_imid, cap=test_cap)
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=4, shuffle=True)

    print("Images are going to be displayed with predicted captions")
    # Some training images and predicted captions are shown
    for i, samples in enumerate(train_loader):
        rnn.eval()
        cnn.eval()
        # Forward Propagation
        sample_ims = samples[0].float().cuda()
        sample_caps = samples[1].to(torch.int64).cuda()
        encoded = cnn(sample_ims)
        decoded = rnn.predict(encoded)
        d = []

        for word in range(17):
            if dictionary[int(torch.argmax(decoded, dim=2)[0, word])] != 'x_END_' and\
                dictionary[int(torch.argmax(decoded, dim=2)[0, word])] != 'x_NULL_' and\
                dictionary[int(torch.argmax(decoded, dim=2)[0, word])] != 'x_START_' and\
                dictionary[int(torch.argmax(decoded, dim=2)[0, word])] != 'x_UNK_':
                d.append(dictionary[int(torch.argmax(decoded, dim=2)[0, word])])

        caption_str = " ".join(d)
        real_im = samples[0].numpy().T.reshape(224, 224, 3)
        plt.imshow(real_im)
        plt.title(caption_str)
        plt.show()
        if i == 10:
            break


    # Some validation images and predicted captions are shown
    for i, samples in enumerate(val_loader):
        rnn.eval()
        cnn.eval()
        # Forward Propagation
        sample_ims = samples[0].float().cuda()
        sample_caps = samples[1].to(torch.int64).cuda()
        encoded = cnn(sample_ims)
        decoded = rnn.predict(encoded)
        d = []

        for word in range(17):
            if dictionary[int(torch.argmax(decoded, dim=2)[0, word])] != 'x_END_' and\
                dictionary[int(torch.argmax(decoded, dim=2)[0, word])] != 'x_NULL_' and\
                dictionary[int(torch.argmax(decoded, dim=2)[0, word])] != 'x_START_' and\
                dictionary[int(torch.argmax(decoded, dim=2)[0, word])] != 'x_UNK_':
                d.append(dictionary[int(torch.argmax(decoded, dim=2)[0, word])])

        caption_str = " ".join(d)
        real_im = samples[0].numpy().T.reshape(224, 224, 3)
        plt.imshow(real_im)
        plt.title(caption_str)
        plt.show()
        if i == 10:
            break


    # Some test images and predicted captions are shown
    for i, samples in enumerate(test_loader):
        rnn.eval()
        cnn.eval()
        # Forward Propagation
        sample_ims = samples[0].float().cuda()
        sample_caps = samples[1].to(torch.int64).cuda()
        encoded = cnn(sample_ims)
        decoded = rnn.predict(encoded)
        d = []

        for word in range(17):
            if dictionary[int(torch.argmax(decoded, dim=2)[0, word])] != 'x_END_' and\
                dictionary[int(torch.argmax(decoded, dim=2)[0, word])] != 'x_NULL_' and\
                dictionary[int(torch.argmax(decoded, dim=2)[0, word])] != 'x_START_' and\
                dictionary[int(torch.argmax(decoded, dim=2)[0, word])] != 'x_UNK_':
                d.append(dictionary[int(torch.argmax(decoded, dim=2)[0, word])])

        caption_str = " ".join(d)
        real_im = samples[0].numpy().T.reshape(224, 224, 3)
        plt.imshow(real_im)
        plt.title(caption_str)
        plt.show()
        if i == 10:
            break





