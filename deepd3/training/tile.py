import numpy as np
import flammkuchen as fl
import cv2
import albumentations as A
import random


class TiledDataGenerator():
    def __init__(self, fn, batch_size, samples_per_epoch=50000, size=(1, 128, 128), target_resolution=None, augment=True,
                 shuffle=True, seed=42, normalize=[-1, 1], min_content=None
                 ):
        """Data Generator that creates tiled data samples from an input image for training DeepD3.
        Essentially like DataGeneratorStream, but the output produced is not a stream, but a list of input tiles.
        Additionally, it only includes significant tiles in the output, with significant meaning that at least
        min_content anotated pixels are available in the correspondent masks.

        Args:
            fn (str): The path to the training data file
            batch_size (int): Batch size for training deep neural networks
            samples_per_epoch (int, optional): Samples used in each epoch. Defaults to 50000.
            size (tuple, optional): Shape of a single sample. Defaults to (1, 128, 128).
            target_resolution (float, optional): Target resolution in microns. Defaults to None.
            augment (bool, optional): Enables augmenting the data. Defaults to True.
            shuffle (bool, optional): Enabled shuffling the data. Defaults to True.
            seed (int, optional): Creates pseudorandom numbers for shuffling. Defaults to 42.
            normalize (list, optional): Values range when normalizing data. Defaults to [-1, 1].
            min_content (float): Hyper-parameter that stablishes the minimum content in image
                                 (annotated dendrite or spine), not considered if 0. Default
                                 to half the tile size in the XY plane.
        """

        # Save settings
        self.batch_size = batch_size
        self.augment = augment
        self.fn = fn
        self.shuffle = shuffle
        self.aug = self._get_augmenter()
        self.seed = seed
        self.normalize = normalize
        self.samples_per_epoch = samples_per_epoch
        self.size = size
        self.target_resolution = target_resolution
        # Default to 30% of image pixels as minimum masked
        self.min_content = float(
            size[1] * size[2] * 0.003) if min_content is None else min_content

        self.d = fl.load(self.fn)
        self.data = self.d['data']
        self.meta = self.d['meta']

        self.cur_tile_x = 0
        self.cur_tile_y = 0

        # Seed randomness
        random.seed(self.seed)
        np.random.seed(self.seed)

        # self.on_epoch_end() TODO: necessary?

    def len_epoch(self):
        """Denotes the number of batches per epoch"""
        return self.samples_per_epoch // self.batch_size

    def get_batch(self, index):
        """Generate one batch of data

        Parameters
        ----------
        index : int
            batch index in image/label id list

        Returns
        -------
        tuple
            Contains two numpy arrays,
            each of shape (batch_size, height, width, 1) = (N, Y, X, Z).
        """
        X = []
        Y0 = []
        Y1 = []
        eps = 1e-5

        if self.shuffle is False:
            np.random.seed(index)

        # Create all pairs in a given batch
        while len(X) < self.batch_size:
            # Retrieve a single sample pair
            image, dendrite, spines = self.get_sample()

            # Augmenting the data
            if self.augment:
                augmented = self.aug(
                    image=image.astype(np.uint8),
                    mask1=dendrite.astype(np.uint8),
                    mask2=spines.astype(np.uint8)
                )
                image = augmented['image']
                dendrite = augmented['mask1']
                spines = augmented['mask2']

            # Min/max scaling
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min() + eps)
            # Shifting and scaling
            image = image * \
                (self.normalize[1]-self.normalize[0]) + self.normalize[0]

            X.append(image)
            Y0.append(dendrite.astype(np.float32) / (dendrite.max() + eps))
            # to ensure binary targets
            Y1.append(spines.astype(np.float32) / (spines.max() + eps))

        return np.asarray(X, dtype=np.float32)[..., None], (np.asarray(Y0, dtype=np.float32)[..., None],
                                                            np.asarray(Y1, dtype=np.float32)[..., None])

    def _get_augmenter(self):
        """Defines used augmentations"""
        aug = A.Compose([
            A.RandomBrightnessContrast(p=0.25),
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(p=0.2),
            A.GaussNoise(p=0.5)], p=1,
            additional_targets={
                'mask1': 'mask',
                'mask2': 'mask'
        })
        return aug

    def get_sample(self, squeeze=True):
        """Get a sample from the provided data with significant segmented masks

        Args:
            squeeze (bool, optional): if plane is 2D, skip 3D. Defaults to True.

        Returns:
            list(np.ndarray, np.ndarray, np.ndarray): stack image with respective labels
        """
        max_tile_y = self.meta.iloc[0].Height - self.size[1] + 1
        max_tile_x = self.meta.iloc[0].Width - self.size[2] + 1
        for y in range(self.cur_tile_y, max_tile_y, self.size[1] // 2):
            start_tile_x = self.cur_tile_x if y == self.cur_tile_y else 0
            for x in range(start_tile_x, max_tile_x, self.size[2] // 2):
                r = self._get_sample(x, y, squeeze)
                if r is None:
                    continue
                # In either one or both annotations there should be at least `min_content` pixels
                # that are labelled
                dend_mask_pixels = (r[1]).sum()
                spine_mask_pixels = (r[2]).sum()
                if dend_mask_pixels > self.min_content or spine_mask_pixels > self.min_content:
                    self.cur_tile_x = x
                    self.cur_tile_y = y
                    return r
        # If we got to the end of the image, go back to the beginning and try again
        self.cur_tile_x = 0
        self.cur_tile_y = 0
        return self.get_sample(squeeze)

    def _get_sample(self, x, y, squeeze=True):
        """Retrieves a sample

        Args:
            squeeze (bool, optional): Squeezes return shape. Defaults to True.
            x = init x position
            y = init y position

        Returns:
            tuple: Tuple of stack (X), dendrite (Y0) and spines (Y1)
        """
        # Adjust for 2 dimensional images
        if len(self.size) == 2:
            size = (1,) + self.size
        else:
            size = self.size

        # sample random stack TODO: not random, sharpest
        r_stack = np.random.choice(len(self.meta))
        meta = self.meta.iloc[r_stack]

        target_h = size[1]
        target_w = size[2]

        if self.target_resolution is None:
            # Keep everything as is
            scaling = 1
        else:
            # Computing scaling factor
            scaling = self.target_resolution / meta.Resolution_XY

        # Compute the height and width and random offsets
        h = round(scaling * target_h)
        w = round(scaling * target_w)

        # Correct for stack dimensions
        if meta.Width-w == 0:
            x = 0
        elif meta.Width-w < 0:
            return

        # Correct for stack dimensions
        if meta.Height-h == 0:
            y = 0
        elif meta.Height-h < 0:
            return

        if x + w > meta.Width or y + h > meta.Height:
            return

        # Select random plane + range
        if (size[0] > meta.Depth-size[0]):
            return

        # TODO: not random
        z_begin = np.random.choice(meta.Depth-size[0]+1)
        z_end = z_begin + size[0]

        # Scale if neccessary to the correct dimensions
        tmp_stack = self.data['stacks'][f'x{r_stack}'][z_begin:z_end, y:y+h, x:x+w]
        tmp_dendrites = self.data['dendrites'][f'x{r_stack}'][z_begin:z_end, y:y+h, x:x+w]
        tmp_spines = self.data['spines'][f'x{r_stack}'][z_begin:z_end, y:y+h, x:x+w]

        # Data needs to be rescaled
        if scaling != 1:
            return_stack = []
            return_dendrites = []
            return_spines = []

            # Do this for each plane
            # and ensure that OpenCV is happy
            for i in range(tmp_stack.shape[0]):
                return_stack.append(cv2.resize(
                    tmp_stack[i], (target_h, target_w)))
                return_dendrites.append(cv2.resize(tmp_dendrites[i].astype(
                    np.uint8), (target_h, target_w)).astype(bool))
                return_spines.append(cv2.resize(tmp_spines[i].astype(
                    np.uint8), (target_h, target_w)).astype(bool))

            return_stack = np.asarray(return_stack)
            return_dendrites = np.asarray(return_dendrites)
            return_spines = np.asarray(return_spines)

        else:
            return_stack = tmp_stack
            return_dendrites = tmp_dendrites
            return_spines = tmp_spines

        if squeeze:
            # Return sample
            return return_stack.squeeze(), return_dendrites.squeeze(), return_spines.squeeze()

        else:
            return return_stack, return_dendrites, return_spines
