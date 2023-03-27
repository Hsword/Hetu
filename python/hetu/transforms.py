from PIL import Image

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for transform in self.transforms:
            img = transform(img)
        return img


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize(self.size, self.interpolation)
        

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        new_width, new_height = self.size
        width, height = img.size  

        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        return img.crop((left, top, right, bottom))


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        new_width, new_height = self.size
        width, height = img.size  

        left = np.random.randint(width - new_width + 1)
        top = np.random.randint(height - new_height + 1)
        right = left + new_width
        bottom = top + new_height

        return img.crop((left, top, right, bottom)) 


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        for i in range(3):
            img[:,:,i] = (img[:,:,i] - self.mean[i]) / self.std[i]
        return img
