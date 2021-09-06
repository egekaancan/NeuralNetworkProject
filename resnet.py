# Libraries are imported
import torchvision
import torch


# CNN class uses pretrained Resnet152 weights to encode the given images
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Resnet152 Initialization
        self.rn = torchvision.models.resnet152(pretrained=True)
        # Last layer is removed
        self.layers = list(self.rn.children())[:-1]
        self.last_layer = list(self.rn.children())[-1]

        # New sequential network is constructed
        self.rn = torch.nn.Sequential(*self.layers)
        # Prohibits change of the pretrained weights of the Resnet152
        for layer in self.rn.parameters():
            layer.requires_grad = False
        # Last fully connected layer is added with output dimension of 300 (which is same as dimemsion of embedding matrix taken from glove)
        self.linear = torch.nn.Linear(self.last_layer.in_features, 300)
        # Batch normalization layer is initiaized
        self.batch_norm = torch.nn.BatchNorm1d(300, momentum=0.01)

    # This part is a basic forward propagation of resnet and batch normalization
    def forward(self, samples):

        resnet_out = self.rn(samples)
        resnet_out_flat = resnet_out.reshape(resnet_out.shape[0], resnet_out.shape[1] * resnet_out.shape[2] * resnet_out.shape[3])
        resnet_out_norm = self.batch_norm(self.linear(resnet_out_flat))
        return resnet_out_norm


















