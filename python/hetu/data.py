import numpy as np
import os


def mnist(dataset='mnist.pkl.gz', onehot=True):
    import six.moves.cPickle as pickle
    import gzip
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix), np.float32
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector), np.int64 that has the same length
    # as the number of rows in the input. It should give the target
    # to the example with the same index in the input.
    if onehot:
        train_set = (train_set[0], convert_to_one_hot(train_set[1], 10))
        valid_set = (valid_set[0], convert_to_one_hot(valid_set[1], 10))
        test_set = (test_set[0], convert_to_one_hot(test_set[1], 10))
    return train_set, valid_set, test_set


def cifar10(directory='CIFAR_10', onehot=True):
    import six.moves.cPickle as pickle
    file_lists = [os.path.join(directory, 'cifar-10-batches-py', 'data_batch_%d' % i) for i in range(1, 6)] +\
        [os.path.join(directory, 'cifar-10-batches-py', 'test_batch')]
    if not all([os.path.exists(fl) for fl in file_lists]):
        from tqdm import tqdm
        from six.moves import urllib
        import tarfile
        filename = "cifar-10-python.tar.gz"
        if not os.path.exists(filename):
            def gen_bar_updater():
                pbar = tqdm(total=None)

                def bar_update(count, block_size, total_size):
                    if pbar.total is None and total_size:
                        pbar.total = total_size
                    progress_bytes = count * block_size
                    pbar.update(progress_bytes - pbar.n)
                return bar_update
            print('Downloading CIFAR 10 dataset...')
            url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            urllib.request.urlretrieve(
                url, filename, reporthook=gen_bar_updater())
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(path=directory)

    images, labels = [], []
    for filename in file_lists[:5]:
        with open(filename, 'rb') as fo:
            cifar10 = pickle.load(fo, encoding='latin1')
        for i in range(len(cifar10["labels"])):
            image = cifar10["data"][i]
            image = image.astype(float)
            images.append(image)
        labels += cifar10["labels"]
    images = np.array(images, dtype='float')
    labels = np.array(labels, dtype='int')
    train_images, train_labels = images, labels

    images, labels = [], []
    for filename in file_lists[5:]:
        with open(filename, 'rb') as fo:
            cifar10 = pickle.load(fo, encoding='latin1')
        for i in range(len(cifar10["labels"])):
            image = cifar10["data"][i]
            image = image.astype(float)
            images.append(image)
        labels += cifar10["labels"]
    images = np.array(images, dtype='float')
    labels = np.array(labels, dtype='int')
    test_images, test_labels = images, labels
    if onehot:
        train_labels = convert_to_one_hot(train_labels, 10)
        test_labels = convert_to_one_hot(test_labels, 10)
    return train_images, train_labels, test_images, test_labels


def cifar100(directory='CIFAR_100', onehot=True):
    import six.moves.cPickle as pickle
    file_lists = [os.path.join(directory, 'cifar-100-python', 'train'),
                  os.path.join(directory, 'cifar-100-python', 'test')]

    if not all([os.path.exists(fl) for fl in file_lists]):
        from tqdm import tqdm
        from six.moves import urllib
        import tarfile
        filename = "cifar-100-python.tar.gz"
        if not os.path.exists(filename):
            def gen_bar_updater():
                pbar = tqdm(total=None)

                def bar_update(count, block_size, total_size):
                    if pbar.total is None and total_size:
                        pbar.total = total_size
                    progress_bytes = count * block_size
                    pbar.update(progress_bytes - pbar.n)
                return bar_update
            print('Downloading CIFAR 100 dataset...')
            url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            urllib.request.urlretrieve(
                url, filename, reporthook=gen_bar_updater())
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(path=directory)

    with open(file_lists[0], 'rb') as input_file:
        train_file = pickle.load(input_file, encoding='latin1')
        train_images = train_file['data']
        train_labels = train_file['fine_labels']
    train_images = np.array(train_images, dtype='float').reshape(
        train_images.shape[0], 3, 32, 32)
    train_labels = np.array(train_labels, dtype='int')

    with open(file_lists[1], 'rb') as input_file:
        test_file = pickle.load(input_file, encoding='latin1')
        test_images = test_file['data']
        test_labels = test_file['fine_labels']
    test_images = np.array(test_images, dtype='float').reshape(
        test_images.shape[0], 3, 32, 32)
    test_labels = np.array(test_labels, dtype='int')

    if onehot:
        train_labels = convert_to_one_hot(train_labels, 100)
        test_labels = convert_to_one_hot(test_labels, 100)

    return train_images, train_labels, test_images, test_labels


def normalize_cifar(num_class=10, onehot=True):
    if num_class == 10:
        x_train, y_train, x_test, y_test = cifar10(onehot=onehot)
    elif num_class == 100:
        x_train, y_train, x_test, y_test = cifar100(onehot=onehot)
    else:
        raise NotImplementedError

    x_train = x_train.reshape((-1, 3, 32, 32))
    x_test = x_test.reshape((-1, 3, 32, 32))
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:, 0, :, :] = (
        x_train[:, 0, :, :] - np.mean(x_train[:, 0, :, :])) / np.std(x_train[:, 0, :, :])
    x_train[:, 1, :, :] = (
        x_train[:, 1, :, :] - np.mean(x_train[:, 1, :, :])) / np.std(x_train[:, 1, :, :])
    x_train[:, 2, :, :] = (
        x_train[:, 2, :, :] - np.mean(x_train[:, 2, :, :])) / np.std(x_train[:, 2, :, :])

    x_test[:, 0, :, :] = (
        x_test[:, 0, :, :] - np.mean(x_test[:, 0, :, :])) / np.std(x_test[:, 0, :, :])
    x_test[:, 1, :, :] = (
        x_test[:, 1, :, :] - np.mean(x_test[:, 1, :, :])) / np.std(x_test[:, 1, :, :])
    x_test[:, 2, :, :] = (
        x_test[:, 2, :, :] - np.mean(x_test[:, 2, :, :])) / np.std(x_test[:, 2, :, :])

    return x_train, y_train, x_test, y_test


def tf_normalize_cifar(num_class=10, onehot=True):
    if num_class == 10:
        x_train, y_train, x_test, y_test = cifar10(onehot=onehot)
    elif num_class == 100:
        x_train, y_train, x_test, y_test = cifar100(onehot=onehot)
    else:
        raise NotImplementedError
    x_train = x_train.reshape((-1, 3, 32, 32))
    x_test = x_test.reshape((-1, 3, 32, 32))
    x_train = x_train.transpose([0, 2, 3, 1]).astype('float32')
    x_test = x_test.transpose([0, 2, 3, 1]).astype('float32')

    x_train[:, :, :, 0] = (
        x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (
        x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (
        x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (
        x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (
        x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (
        x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, y_train, x_test, y_test


def convert_to_one_hot(vals, max_val=0):
    """Helper method to convert label array to one-hot array."""
    if max_val == 0:
        max_val = vals.max() + 1
    one_hot_vals = np.zeros((vals.size, max_val))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals


########################
# Not in use currently #
########################

def data_augmentation(images, mode='train', flip=False,
                      crop=False, crop_shape=(24, 24, 3), whiten=False,
                      noise=False, noise_mean=0, noise_std=0.01):
    if crop:
        if mode == 'train':
            images = self._image_crop(images, shape=crop_shape)
        elif mode == 'test':
            images = self._image_crop_test(images, shape=crop_shape)
    if flip:
        images = self._image_flip(images)
    if whiten:
        images = self._image_whitening(images)
    if noise:
        images = self._image_noise(images, mean=noise_mean, std=noise_std)

    return images


def _image_crop(images, shape):
    new_images = []
    for i in range(images.shape[0]):
        old_image = images[i, :, :, :]
        old_image = numpy.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
        left = numpy.random.randint(old_image.shape[0] - shape[0] + 1)
        top = numpy.random.randint(old_image.shape[1] - shape[1] + 1)
        new_image = old_image[left: left+shape[0], top: top+shape[1], :]
        new_images.append(new_image)

    return numpy.array(new_images)


def _image_crop_test(images, shape):
    new_images = []
    for i in range(images.shape[0]):
        old_image = images[i, :, :, :]
        old_image = numpy.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
        left = int((old_image.shape[0] - shape[0]) / 2)
        top = int((old_image.shape[1] - shape[1]) / 2)
        new_image = old_image[left: left+shape[0], top: top+shape[1], :]
        new_images.append(new_image)

    return numpy.array(new_images)


def _image_flip(images):
    for i in range(images.shape[0]):
        old_image = images[i, :, :, :]
        if numpy.random.random() < 0.5:
            new_image = cv2.flip(old_image, 1)
        else:
            new_image = old_image
        images[i, :, :, :] = new_image

    return images


def _image_whitening(images):
    for i in range(images.shape[0]):
        old_image = images[i, :, :, :]
        new_image = (old_image - numpy.mean(old_image)) / numpy.std(old_image)
        images[i, :, :, :] = new_image

    return images


def _image_noise(images, mean=0, std=0.01):
    for i in range(images.shape[0]):
        old_image = images[i, :, :, :]
        new_image = old_image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    new_image[i, j, k] += random.gauss(mean, std)
        images[i, :, :, :] = new_image

    return images
