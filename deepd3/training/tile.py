# Local imports
from utils.stack import Stack

# External libraries imports
import numpy as np
import flammkuchen as fl
import cv2
import albumentations as A
import random
import tifffile
from typing import List
from collections import namedtuple
from tensorflow.keras.utils import Sequence


class TiledDataGenerator(Sequence):

    def __init__(
        self,
        batch_size,
        fn=List[Stack],
        samples_per_epoch=50000,
        size=(1, 128, 128),
        target_resolution=None,
        augment=True,
        shuffle=True,
        seed=42,
        normalize=[-1, 1],
        min_content=None
    ):
        """Data Generator that creates tiled data samples from an input image for training DeepD3.
        Essentially like DataGeneratorStream, but the output produced is not a stream, but a list of input tiles.
        Additionally, it only includes significant tiles in the output, with significant meaning that at least
        min_content anotated pixels are available in the correspondent masks.

        Args:
            fn (str): A list of paths to the data stacks. If only one element is provided, it should contain only a path for img for the d3set.
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
        if len(size) == 2:
            # Adjust for 2 dimensional images
            self.size = (1,) + size
        else:
            self.size = size
        self.target_z = self.size[0]
        self.target_h = self.size[1]
        self.target_w = self.size[2]
        self.target_resolution = target_resolution
        self.steps_per_epoch = self.samples_per_epoch // self.batch_size
        # Default to 30% of image pixels as minimum masked
        self.min_content = float(
            size[1] * size[2] * 0.003) if min_content is None else min_content

        # Load data
        if ('meta' in fn[0]):
            self.load_raw()
        else:
            self.load_d3set()

        self.batch_index = 0
        self.epoch_index = 0

        # Seed randomness
        random.seed(self.seed)
        np.random.seed(self.seed)

    def load_d3set(self):
        d = fl.load(self.fn[0]['img'])
        # stacks, dendrite, spines: [n_stacks][z,y,x]
        self.data = d['data']
        # [n_stacks]{Height,Width,Depth,Resolution_XY,Resolution_Z}
        self.meta = d['meta'].iloc
        self.n_stacks = len(d['meta'])

    def load_raw(self):
        # Load raw images as np arrays
        self.data = {'stacks': {}, "dendrites": {}, "spines": {}}
        self.meta = []
        for i in range(len(self.fn)):
            img = tifffile.imread(self.fn[i]['img'])
            d_mask = tifffile.imread(self.fn[i]['d_mask'])
            s_masks = tifffile.imread(self.fn[i]['s_masks'])
            # Original images are stored as (Z, X, Y)
            img = np.transpose(img, axes=(0, 2, 1))
            d_mask = np.transpose(d_mask, axes=(0, 2, 1))
            s_masks = np.transpose(s_masks, axes=(0, 2, 1))
            # Load metadata following d3set format
            MetaEntry = namedtuple(
                'MetaEntry', ['Height', 'Width', 'Depth', 'Resolution_XY', 'Resolution_Z'])
            z, x, y = self.fn[i]['meta']['stack_shapes']['sizes']
            res_z, res_x, res_y = self.fn[i]['meta']['pixel_sizes']['sizes']
            meta_entry = MetaEntry(
                Height=y,
                Width=x,
                Depth=z,
                # Resolutions are in m, so we convert them to um (micrometer)
                Resolution_XY=max(res_x * 1e+6, res_y * 1e+6),
                Resolution_Z=res_z * 1e+6
            )
            self.meta.append(meta_entry)

            # Assemble everything
            self.data['stacks'][f"x{i}"] = img
            self.data['dendrites'][f"x{i}"] = d_mask
            self.data['spines'][f"x{i}"] = s_masks
        self.n_stacks = len(self.meta)

    def on_epoch_end(self):
        self.batch_index = 0
        self.epoch_index += 1
        random.seed(self.seed + self.epoch_index)
        np.random.seed(self.seed + self.epoch_index)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.steps_per_epoch

    # def __iter__(self):
    #     self.on_epoch_end()
    #     return self

    # def __next__(self):
    #     if self.batch_index >= self.__len__():
    #         # print(f"raising stop in batch index {self.batch_index} with len {self.__len__()} and batch_size {self.batch_size},  self.samples_per_epoch = { self.samples_per_epoch} and  self.steps_per_epoch = {self.steps_per_epoch}")
    #         self.on_epoch_end()
    #         raise StopIteration
    #     # print(f"updating {self.batch_index} to {self.batch_index+1}")
    #     self.batch_index += 1
    #     return self.__getitem__(self.batch_index)

    def __getitem__(self, index):
        """Generate one batch of data

        Args:
            index (int): Batch index in image/label id list.

        Returns:
            tuple: Contains two numpy arrays, each of shape (batch_size, 1, height, width) = (N, Z, Y, X).
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
        """Get a sample from one of the available stacks with significant segmentation masks

        Args:
            squeeze (bool, optional): if plane is 2D, skip 3D. Defaults to True.

        Returns:
            list(np.ndarray, np.ndarray, np.ndarray): stack image with respective labels
        """
        # Select random stack
        r_stack = np.random.choice(self.n_stacks)
        meta = self.meta[r_stack]

        # Compute the height and width of tiles (accounting for scaling)
        if self.target_resolution is None:
            scaling = 1
        else:
            scaling = self.target_resolution / meta.Resolution_XY
        h = round(scaling * self.target_h)
        w = round(scaling * self.target_w)

        # Iterate through possible coordinates
        max_tile_y = meta.Height - self.target_h + 1
        max_tile_x = meta.Width - self.target_w + 1
        coords = [
            (y, x) for y in range(0, max_tile_y, h) for x in range(0, max_tile_x, w)
        ]
        np.random.shuffle(coords)
        for coord in coords:
            y, x = coord
            r = self._get_sample(meta, r_stack, y, x, h, w, scaling, squeeze)
            if r is None:
                continue
            # In either one or both annotations there should be at least `min_content` pixels
            # that are labelled
            dend_mask_pixels = (r[1]).sum()
            spine_mask_pixels = (r[2]).sum()
            if (
                dend_mask_pixels > self.min_content
                or spine_mask_pixels > self.min_content
            ):
                return r
        # If no sample was found in this stack, try another one
        return self.get_sample(squeeze)

    def _get_sample(self, meta, r_stack, y, x, h, w, scaling, squeeze=True):
        """Retrieves a sample

        Args:
            squeeze (bool, optional): Squeezes return shape. Defaults to True.
            x = init x position
            y = init y position

        Returns:
            tuple: Tuple of stack (X), dendrite (Y0) and spines (Y1)
        """
        # Correct x coordinate for stack dimensions
        if meta.Width-w == 0:
            x = 0
        elif meta.Width-w < 0:
            return

        # Correct y coordinate for stack dimensions
        if meta.Height-h == 0:
            y = 0
        elif meta.Height-h < 0:
            return

        if x + w > meta.Width or y + h > meta.Height:
            return

        # Select random plane + range, if possible
        if self.target_z > meta.Depth - self.target_z:
            return
        z_begin = np.random.choice(meta.Depth - self.target_z + 1)
        z_end = z_begin + self.target_z

        # Get stack, spines and dendrites
        stack_data = self.data["stacks"][f"x{r_stack}"]
        dendrite_mask_data = self.data["dendrites"][f"x{r_stack}"]
        spines_mask_data = self.data["spines"][f"x{r_stack}"]

        # Scale if necessary to the correct dimensions
        tmp_stack = stack_data[z_begin:z_end, y: y + h, x: x + w]
        tmp_dendrites = dendrite_mask_data[z_begin:z_end, y: y + h, x: x + w]
        tmp_spines = spines_mask_data[z_begin:z_end, y: y + h, x: x + w]

        # Sanity checks
        assert tmp_stack.shape == (
            self.size[0],
            h,
            w,
        ), "Unexpected orig shape: {tmp_stack.shape} at x={x}, y={y}"
        assert tmp_dendrites.shape == (
            self.size[0],
            h,
            w,
        ), "Unexpected dendrite shape: {tmp_dendrites.shape} at x={x}, y={y}"
        assert tmp_spines.shape == (
            self.size[0],
            h,
            w,
        ), "Unexpected spines shape: {tmp_spines.shape} at x={x}, y={y}"

        # Data needs to be rescaled
        if scaling != 1:
            return_stack = []
            return_dendrites = []
            return_spines = []

            # Do this for each plane
            # and ensure that OpenCV is happy
            for i in range(tmp_stack.shape[0]):
                return_stack.append(
                    cv2.resize(tmp_stack[i], (self.target_h, self.target_w))
                )
                return_dendrites.append(
                    cv2.resize(
                        tmp_dendrites[i].astype(np.uint8),
                        (self.target_h, self.target_w),
                    ).astype(bool)
                )
                return_spines.append(
                    cv2.resize(
                        tmp_spines[i].astype(np.uint8), (self.target_h, self.target_w)
                    ).astype(bool)
                )

            return_stack = np.asarray(return_stack)
            return_dendrites = np.asarray(return_dendrites)
            return_spines = np.asarray(return_spines)

        else:
            return_stack = tmp_stack
            return_dendrites = tmp_dendrites
            return_spines = tmp_spines

        if squeeze:
            return return_stack.squeeze(), return_dendrites.squeeze(), return_spines.squeeze()

        else:
            return return_stack, return_dendrites, return_spines
