import glob
import torch
import pdb
import os
import numbers
import numpy as np
import math
import nibabel as nib
import PIL
import cv2
import random
import collections
import torch.utils.data
import torchvision
import json
import torchvision.transforms as transforms
try:
    import accimage
except ImportError:
    accimage = None
import time
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from torch.utils.data.distributed import DistributedSampler
import config


def data_loader(args, mode):
    # Data Flag Check

    # Mode Flag Check 
    if mode == 'train':
        shuffle = True
        dataset=TrainSet(args)
        sampler = DistributedSampler(dataset)
        batch_sampler = torch.utils.data.BatchSampler(
        sampler, args.batch_size, drop_last=True)
    elif mode == 'valid':
        shuffle = False
        dataset = ValidSet(args)
        sampler = DistributedSampler(dataset)
        batch_sampler = torch.utils.data.BatchSampler(
        sampler, int(args.batch_size/args.world_size), drop_last=True)
    elif mode == 'test':
        shuffle = False
        dataset = TestSet(args)
        sampler = DistributedSampler(dataset)
        batch_sampler = torch.utils.data.BatchSampler(
        sampler, int(args.batch_size/args.world_size), drop_last=True)
    else:
        raise ValueError('data_loader mode ERROR')
    nw = min([os.cpu_count(),args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler,
                                   num_workers=nw, pin_memory=True)
    return dataloader,sampler


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, data_list, prefix='../../data/sd0809/BraTS2020', transform=None):
        json_path = 'brats20.json'
        data_list = self.load_json(json_path)
        self.data_list = data_list['train']
        self.prefix = prefix
        self.transform = transform

        # 加载所有数据
        self.images = []
        self.labels = []
        self.slices = []
        i = 0
        for data_item in self.data_list:
            selected_images = []
            for image_path in data_item['image']:
                if 't1c' in image_path or 't2' in image_path or 'flair' in image_path:
                    selected_images.append(image_path)
            
            images = [nib.load(os.path.join(self.prefix, image_path)).get_fdata()[:, :, 79:129] for image_path in selected_images]
            label = nib.load(os.path.join(self.prefix, data_item['label'])).get_fdata()[:, :, 79:129]
            
            # 确保所有模态和标签的形状一致
            assert all(image.shape == label.shape for image in images)
            print(data_item['label'])
            # 体积归一化
            images = [self.normalize_volume(image) for image in images]
            
            self.images.append(images)
            self.labels.append(label)
            
            for slice_idx in range(50):  # 切片索引从0到49
                self.slices.append((i, slice_idx))
            
            i += 1
            if i==50:
                break
        random.shuffle(self.slices)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_idx, slice_idx = self.slices[idx]
        images = [img[:, :, slice_idx] for img in self.images[img_idx]]
        label = self.labels[img_idx][:, :, slice_idx]

        # Stack images along the channel dimension
        slice_images = np.stack(images, axis=0)

        # Optionally apply transformations
        if self.transform:
            slice_images, slice_label = self.transform(slice_images, label)

        # Convert to torch tensors
        slice_images = torch.tensor(slice_images, dtype=torch.float32).permute(0,2,1)
        slice_label = torch.tensor(label, dtype=torch.long).permute(1,0)
        #import pdb;pdb.set_trace()
        slice_label[slice_label == 4] = 3

        return slice_images, slice_label

    def normalize_volume(self, volume):
        """归一化体积数据到 [0, 1]"""
        volume = volume.astype(np.float32)
        min_val = np.min(volume)
        max_val = np.max(volume)
        normalized_volume = (volume - min_val) / (max_val - min_val + 1e-5)  # 加上 1e-5 以防止除零
        return normalized_volume

    def load_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data



class TestSet(torch.utils.data.Dataset):
    def __init__(self, data_list, prefix='../../data/sd0809/BraTS2020', transform=None):
        json_path = 'brats20.json'
        data_list = self.load_json(json_path)
        self.data_list = data_list['test']
        self.prefix = prefix
        self.transform = transform

        # 加载所有数据并分割成单个切片
        self.slices = []
        for data_item in self.data_list:
            selected_images = []
            for image_path in data_item['image']:
                if 't1c' in image_path or 't2' in image_path or 'flair' in image_path:
                    selected_images.append(image_path)
            
            images = [nib.load(os.path.join(self.prefix, image_path)).get_fdata()[:, :, 79:129] for image_path in selected_images]
            label = nib.load(os.path.join(self.prefix, data_item['label'])).get_fdata()[:, :, 79:129]

            # 确保所有模态和标签的形状一致
            assert all(image.shape == label.shape for image in images)

            # 体积归一化
            images = [self.normalize_volume(image) for image in images]
            print(data_item['image'][0])
            # 按切片分割
            for slice_idx in range(50):  # 假设切片在轴2
                slice_images = [image[:, :, slice_idx] for image in images]
                
                slice_label = label[:, :, slice_idx]
                self.slices.append((slice_images, slice_label))
            

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice_images, slice_label = self.slices[idx]
        
        # Stack images along the channel dimension
        slice_images = np.stack(slice_images, axis=0)
        

        # Optionally apply transformations
        if self.transform:
            slice_images, slice_label = self.transform(slice_images, slice_label)

        # Convert to torch tensors
        #import pdb;pdb.set_trace()
        slice_images = torch.tensor(slice_images, dtype=torch.float32).transpose(1,2)
        slice_label = torch.tensor(slice_label, dtype=torch.long).transpose(0, 1)
        slice_label[slice_label == 4] = 3
        return slice_images, slice_label

    def normalize_volume(self, volume):
        """归一化体积数据到 [0, 1]"""
        volume = volume.astype(np.float32)
        min_val = np.min(volume)
        max_val = np.max(volume)
        normalized_volume = (volume - min_val) / (max_val - min_val + 1e-5)  # 加上 1e-5 以防止除零
        return normalized_volume

    def load_json(self,json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data



class ValidSet(torch.utils.data.Dataset):
    def __init__(self, data_list, prefix='../../data/sd0809/BraTS2020', transform=None):
        json_path = 'brats20.json'
        data_list = self.load_json(json_path)
        self.data_list = data_list['valid']
        self.prefix = prefix
        self.transform = transform

        # 加载所有数据并分割成单个切片
        self.slices = []
        for data_item in self.data_list:
            selected_images = []
            for image_path in data_item['image']:
                if 't1c' in image_path or 't2' in image_path or 'flair' in image_path:
                    selected_images.append(image_path)
            
            images = [nib.load(os.path.join(self.prefix, image_path)).get_fdata()[:, :, 79:129] for image_path in selected_images]
            label = nib.load(os.path.join(self.prefix, data_item['label'])).get_fdata()[:, :, 79:129]

            # 确保所有模态和标签的形状一致
            assert all(image.shape == label.shape for image in images)

            # 体积归一化
            images = [self.normalize_volume(image) for image in images]

            # 按切片分割
            for slice_idx in range(50):  # 假设切片在轴2
                slice_images = [image[:, :, slice_idx] for image in images]
                print(data_item['image'][0], slice_idx)
                slice_label = label[:, :, slice_idx]
                self.slices.append((slice_images, slice_label))
            

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice_images, slice_label = self.slices[idx]
        
        # Stack images along the channel dimension
        slice_images = np.stack(slice_images, axis=0)
        

        # Optionally apply transformations
        if self.transform:
            slice_images, slice_label = self.transform(slice_images, slice_label)

        # Convert to torch tensors
        #import pdb;pdb.set_trace()
        slice_images = torch.tensor(slice_images, dtype=torch.float32).transpose(1,2)
        slice_label = torch.tensor(slice_label, dtype=torch.long).transpose(0, 1)
        slice_label[slice_label == 4] = 3
        return slice_images, slice_label

    def normalize_volume(self, volume):
        """归一化体积数据到 [0, 1]"""
        volume = volume.astype(np.float32)
        min_val = np.min(volume)
        max_val = np.max(volume)
        normalized_volume = (volume - min_val) / (max_val - min_val + 1e-5)  # 加上 1e-5 以防止除零
        return normalized_volume

    def load_json(self,json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data





#############################################################
#                                                           #
#       Data Transforms Functions                           # 
#                                                           #
#############################################################

'''
    From torchvision Transforms.py (+ Slightly changed)
    (https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py)
'''
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

    def __call__(self, img, label):
        for t in self.transforms:
            img,label = t(img, label)
        return img, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic), to_tensor(label)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return vflip(img), vflip(label)
        return img, label

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return hflip(img), hflip(label)
        return img, label

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        #angle = random.uniform(degrees[0], degrees[1])
        angle_list = [0,90,180,270]
        angle = random.choice(angle_list)
        return angle

    def __call__(self, img, label):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        #angle = self.get_params(self.degrees)
        angle = np.random.randint(self.degrees[0], self.degrees[1])
        return rotate(img, angle, self.resample, self.expand, self.center),\
                rotate(label, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img, label):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        return affine(img, label, *ret, resample=self.resample, fillcolor=self.fillcolor)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)





'''
    From torchvision functional.py (+ Slightly changed)
    (https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py)
'''
def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, P, I, F, RGB, YCbCr, RGBA, CMYK
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
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img

def vflip(img):
    """Vertically flip the given PIL Image.
    Args:
        img (PIL Image): Image to be flipped.
    Returns:
        PIL Image:  Vertically flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_TOP_BOTTOM)

def hflip(img):
    """Horizontally flip the given PIL Image.
    Args:
        img (PIL Image): Image to be flipped.
    Returns:
        PIL Image:  Horizontall flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_LEFT_RIGHT)

def rotate(img, angle, resample=False, expand=False, center=None):
    """Rotate the image by angle.
    Args:
        img (PIL Image): PIL Image to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.rotate(angle, resample, expand, center)


def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
    #                              [ sin(a)*scale    cos(a + shear)*scale     0]
    #                              [     0                  0          1]
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    angle = math.radians(angle)
    shear = math.radians(shear)
    scale = 1.0 / scale

    # Inverted rotation matrix with scale and shear
    d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
    matrix = [
        math.cos(angle + shear), math.sin(angle + shear), 0,
        -math.sin(angle), math.cos(angle), 0
    ]
    matrix = [scale / d * m for m in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
    matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]
    return matrix


def affine(img, label, angle, translate, scale, shear, resample=0, fillcolor=None):
    """Apply affine transformation on the image keeping image center invariant
    Args:
        img (PIL Image): PIL Image to be rotated.
        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations(post-rotation translation)
        scale (float): overall scale
        shear (float): shear angle value in degrees between -180 to 180, clockwise direction.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter.
            See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] == '5' else {}
    return img.transform(output_size, Image.AFFINE, matrix, resample, **kwargs),\
            label.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)




"""
	source code From.
	https://gist.github.com/oeway/2e3b989e0343f0884388ed7ed82eb3b0
"""

class ElasticTransform(object):
    """Apply elastic transformation on a numpy.ndarray (H x W x C)
    """

    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image, label):
        if isinstance(self.alpha, collections.Sequence):
            alpha = random_num_generator(self.alpha)
        else:
            alpha = self.alpha
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(self.sigma)
        else:
            sigma = self.sigma
        return elastic_transform(image, label, alpha=alpha, sigma=sigma)


def elastic_transform(image, label, alpha=1000, sigma=30, spline_order=1, mode='nearest', random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    #assert image.ndim == 3
    image = np.array(image)
    label = np.array(label)
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]

    result1 = map_coordinates(image, indices, order=spline_order, mode=mode).reshape(shape)
    result2 = map_coordinates(label, indices, order=spline_order, mode=mode).reshape(shape)
    return Image.fromarray(result1), Image.fromarray(result2)


def random_num_generator(config, random_state=np.random):
    if config[0] == 'uniform':
        ret = random_state.uniform(config[1], config[2], 1)[0]
    elif config[0] == 'lognormal':
        ret = random_state.lognormal(config[1], config[2], 1)[0]
    else:
        print(config)
        raise Exception('unsupported format')
    return ret
