# Libraries are imported
from torch.utils.data import Dataset
import numpy as np
from skimage import io
from skimage.transform import resize
import cv2
from torchvision import transforms


# ImageData inherits Dataset class of torch library
# This class will be used to extract the images from jpg file into ram locations
class ImageData(Dataset):

    # directory is the folder that jpg files are stored
    # imid is image ids that represents the number of image that we want to use
    # cap is captions of the images
    def __init__(self, directory, imid, cap):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.directory = directory
        self.imid = imid
        self.cap = cap
        self.length = self.imid.shape[0]
        self.temp_cap = self.cap[10]
        self.temp_im = io.imread(self.directory + '/' + str(self.imid[10]-1) + '.jpg')

    # Allow us to select an item
    # This overwritten function will be used by Dataloder class
    # returns image and caption of given imid
    def __getitem__(self, item):
        try:
            # Data is read from the directory
            im = io.imread(self.directory + '/' + str(self.imid[item]-1) + '.jpg')
            # In case there is an grayscale image, image is transformed into rgb
            if len(im.shape) == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            # image is resized with shape (224, 224, 3) since RESNET takes images with 224x224x3 pixel square images
            im = resize(image=im, output_shape=(224, 224))
            # If all these parts are passed by a problematic image is encountered, class returns previous image and its caption
            if im.shape[0] < 100:
                return self.transform(np.array(self.temp_im).copy()), np.array(self.temp_cap).copy()
            # Return of image and caption
            self.temp_cap = self.cap[item]
            self.temp_im = im
            return self.transform(np.array(im).copy()), np.array(self.cap[item]).copy()
        except:
            # If there is no such file that imid wants, return previous image and caption
            return self.transform(np.array(self.temp_im).copy()), np.array(self.temp_cap).copy()

    # Length of the dataset that will be used by Dataloader to divide dataset into batches
    def __len__(self):
        return self.length














