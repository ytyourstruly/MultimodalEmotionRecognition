'''
Parts of this code are based on https://github.com/okankop/Efficient-3DCNNs
'''

import random
import numbers
import numpy as np
import torch
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        #print(img.size(), img.float().div(self.norm_value)) 
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass




class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        pass


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self):
        self.p = random.random()


class RandomRotate(object):

    def __init__(self):
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        ret_img = img.rotate(self.rotate_angle, resample=self.interpolation)

        return ret_img

    def randomize_parameters(self):
        self.rotate_angle = random.randint(-10, 10)

class NormalizeAudio(object):
    """Scale waveform to [-1, 1] by peak normalisation.

    Applied in both train and val pipelines — this is not augmentation,
    it is a required pre-processing step so that downstream noise levels
    and model weights are scale-invariant across speakers.
    """

    def __call__(self, y: np.ndarray) -> np.ndarray:
        m = np.max(np.abs(y))
        return (y / m).astype(y.dtype) if m > 0 else y

    def randomize_parameters(self):
        pass
class RandomTimeShift(object):
    def __init__(self, max_shift_ratio: float = 0.1):
        self.max_shift_ratio = max_shift_ratio

    def __call__(self, y: np.ndarray) -> np.ndarray:
        # T is always the last axis whether y is (T,) or (F, T)
        t     = y.shape[-1]
        shift = int(random.uniform(-self.max_shift_ratio, self.max_shift_ratio) * t)

        if shift == 0:
            return y

        # np.roll works on any ndim and any axis
        shifted = np.roll(y, shift, axis=-1)

        # Zero-pad the wrapped region so it isn't circular
        if shift > 0:
            shifted[..., :shift] = 0      # zero leading samples
        else:
            shifted[..., shift:] = 0      # zero trailing samples

        return shifted

    def randomize_parameters(self):
        pass
class RandomNoiseInjection(object):
    def __init__(self, min_snr_db: float = 20.0, max_snr_db: float = 40.0):
        assert min_snr_db <= max_snr_db
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

    def __call__(self, y: np.ndarray) -> np.ndarray:
        signal_rms     = np.sqrt(np.mean(y ** 2)) + 1e-9
        noise_std      = signal_rms / (10 ** (self.snr_db / 20.0))
        noise          = np.random.randn(*y.shape).astype(y.dtype)
        return (y + noise * noise_std).astype(y.dtype)

    def randomize_parameters(self):
        self.snr_db = random.uniform(self.min_snr_db, self.max_snr_db)

class RandomGainVariation(object):
    """Multiply waveform by a random gain factor.

    Simulates per-speaker volume differences that peak normalisation alone
    doesn't fully remove (e.g. recording level, mic distance).

    Args:
        min_gain: lower bound of the uniform gain range (default 0.7)
        max_gain: upper bound of the uniform gain range (default 1.3)
    """

    def __init__(self, min_gain: float = 0.7, max_gain: float = 1.3):
        assert 0 < min_gain <= max_gain, "Need 0 < min_gain <= max_gain"
        self.min_gain = min_gain
        self.max_gain = max_gain

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return (y * self.gain).astype(y.dtype)

    def randomize_parameters(self):
        self.gain = random.uniform(self.min_gain, self.max_gain)