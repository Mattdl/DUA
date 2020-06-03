""" Distortions from: https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_tinyimagenet_c.py """
import numpy as np
from io import BytesIO
import ctypes
import warnings

import torchvision.transforms.functional as TF
from PIL import Image as PILImage

import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage.interpolation import map_coordinates
import cv2

warnings.simplefilter("ignore")


# UTILS
def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# NOISE
class NoiseTransform(object):
    types = ['gaus', 'shot', 'impulse', 'speckle', 'jpeg', 'pixel']

    def __init__(self, type, severity):
        assert type in self.types
        self.type = type
        self.severity = severity
        self.transform = None
        self.parse_type(type)

    def parse_type(self, type):
        if type == 'gaus':
            self.transform = self.gaussian_noise
        elif type == 'shot':
            self.transform = self.shot_noise
        elif type == 'impulse':
            self.transform = self.impulse_noise
        elif type == 'speckle':
            self.transform = self.speckle_noise
        elif type == 'jpeg':
            self.transform = self.jpeg_compression
        elif type == 'pixel':
            self.transform = self.pixelate
        else:
            raise Exception("Not existing type in transform: {}".format(type))

    def __call__(self, sample):
        ret = self.transform(sample, severity=self.severity)
        ret = np.array(ret)  # fix JpegImageFile
        ret = TF.to_pil_image(ret.astype('uint8'))
        return ret

    def gaussian_noise(self, x, severity=1):
        c = [0.04, 0.08, .12, .15, .18][severity - 1]

        x = np.array(x) / 255.
        return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    def shot_noise(self, x, severity=1):
        c = [250, 100, 50, 30, 15][severity - 1]

        x = np.array(x) / 255.
        return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255

    def impulse_noise(self, x, severity=1):
        c = [.01, .02, .05, .08, .14][severity - 1]

        x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
        return np.clip(x, 0, 1) * 255

    def speckle_noise(self, x, severity=1):
        c = [.15, .2, 0.25, 0.3, 0.35][severity - 1]

        x = np.array(x) / 255.
        return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    def jpeg_compression(self, x, severity=1):
        c = [65, 58, 50, 40, 25][severity - 1]

        output = BytesIO()
        x.save(output, 'JPEG', quality=c)
        x = PILImage.open(output)
        return x

    def pixelate(self, x, severity=1):
        c = [0.9, 0.8, 0.7, 0.6, 0.5][severity - 1]

        x = x.resize((int(64 * c), int(64 * c)), PILImage.BOX)
        x = x.resize((64, 64), PILImage.BOX)
        return x


# BLUR
class BlurTransform(object):
    types = ['gaus', 'defocus', 'motion']

    def __init__(self, type, severity):
        assert type in self.types
        self.type = type
        self.severity = severity
        self.transform = None
        self.parse_type(type)

    def parse_type(self, type):
        if type == 'gaus':
            self.transform = self.gaussian_blur
        elif type == 'defocus':
            self.transform = self.defocus_blur
        elif type == 'motion':
            self.transform = self.motion_blur
        else:
            raise Exception("Not existing type in transform: {}".format(type))

    def __call__(self, sample):
        ret = self.transform(sample, severity=self.severity)
        ret = TF.to_pil_image(ret.astype('uint8'))
        return ret

    def gaussian_blur(self, x, severity=1):
        c = [.5, .75, 1, 1.25, 1.5][severity - 1]

        x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
        return np.clip(x, 0, 1) * 255

    def defocus_blur(self, x, severity=1):
        c = [(0.5, 0.6), (1, 0.1), (1.5, 0.1), (2.5, 0.01), (3, 0.1)][severity - 1]

        x = np.array(x) / 255.
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x64x64 -> 64x64x3

        return np.clip(channels, 0, 1) * 255

    def motion_blur(self, x, severity=1):
        from wand.image import Image as WandImage
        from wand.api import library as wandlibrary

        # Tell Python about the C method
        wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                                      ctypes.c_double,  # radius
                                                      ctypes.c_double,  # sigma
                                                      ctypes.c_double)  # angle

        # Extend wand.image.Image class to include method signature
        class MotionImage(WandImage):
            def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
                wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

        c = [(10, 1), (10, 1.5), (10, 2), (10, 2.5), (12, 3)][severity - 1]

        output = BytesIO()
        x.save(output, format='PNG')
        x = MotionImage(blob=output.getvalue())

        x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

        x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                         cv2.IMREAD_UNCHANGED)

        if x.shape != (64, 64):
            return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
        else:  # greyscale to RGB
            return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


# Color
class ColorTransform(object):
    types = ['contrast', 'bright', 'sat']

    def __init__(self, type, severity):
        assert type in self.types
        self.type = type
        self.severity = severity
        self.transform = None
        self.parse_type(type)

    def parse_type(self, type):
        if type == 'contrast':
            self.transform = self.contrast
        elif type == 'bright':
            self.transform = self.brightness
        elif type == 'sat':
            self.transform = self.saturate
        else:
            raise Exception("Not existing type in transform: {}".format(type))

    def __call__(self, sample):
        ret = self.transform(sample, severity=self.severity)
        ret = TF.to_pil_image(ret.astype('uint8'))
        return ret

    def contrast(self, x, severity=1):
        c = [.4, .3, .2, .1, 0.05][severity - 1]

        x = np.array(x) / 255.
        means = np.mean(x, axis=(0, 1), keepdims=True)
        return np.clip((x - means) * c + means, 0, 1) * 255

    def brightness(self, x, severity=1):
        c = [.1, .2, .3, .4, .5][severity - 1]

        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)

        return np.clip(x, 0, 1) * 255

    def saturate(self, x, severity=1):
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (30, 0.2)][severity - 1]

        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
        x = sk.color.hsv2rgb(x)

        return np.clip(x, 0, 1) * 255


# DEFORM
class DeformTransform(object):
    types = ['spatter', 'elastic']

    def __init__(self, type, severity):
        assert type in self.types
        self.type = type
        self.severity = severity
        self.transform = None
        self.parse_type(type)

    def parse_type(self, type):
        if type == 'spatter':
            self.transform = self.spatter
        elif type == 'elastic':
            self.transform = self.elastic_transform
        else:
            raise Exception("Not existing type in transform: {}".format(type))

    def __call__(self, sample):
        ret = self.transform(sample, severity=self.severity)
        ret = TF.to_pil_image(ret.astype('uint8'))
        return ret

    def spatter(self, x, severity=1):
        c = [(0.62, 0.1, 0.7, 0.7, 0.6, 0),
             (0.65, 0.1, 0.8, 0.7, 0.6, 0),
             (0.65, 0.3, 1, 0.69, 0.6, 0),
             (0.65, 0.1, 0.7, 0.68, 0.6, 1),
             (0.65, 0.1, 0.5, 0.67, 0.6, 1)][severity - 1]
        x = np.array(x, dtype=np.float32) / 255.

        liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

        liquid_layer = gaussian(liquid_layer, sigma=c[2])
        liquid_layer[liquid_layer < c[3]] = 0
        if c[5] == 0:
            liquid_layer = (liquid_layer * 255).astype(np.uint8)
            dist = 255 - cv2.Canny(liquid_layer, 50, 150)
            dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
            _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
            dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
            dist = cv2.equalizeHist(dist)
            #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
            #     ker -= np.mean(ker)
            ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            dist = cv2.filter2D(dist, cv2.CV_8U, ker)
            dist = cv2.blur(dist, (3, 3)).astype(np.float32)

            m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
            m /= np.max(m, axis=(0, 1))
            m *= c[4]

            # water is pale turqouise
            color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1])), axis=2)

            color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

            return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
        else:
            m = np.where(liquid_layer > c[3], 1, 0)
            m = gaussian(m.astype(np.float32), sigma=c[4])
            m[m < 0.8] = 0
            #         m = np.abs(m) ** (1/c[4])

            # mud brown
            color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                    42 / 255. * np.ones_like(x[..., :1]),
                                    20 / 255. * np.ones_like(x[..., :1])), axis=2)

            color *= m[..., np.newaxis]
            x *= (1 - m[..., np.newaxis])

            return np.clip(x + color, 0, 1) * 255

    # mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
    def elastic_transform(self, image, severity=1):
        IMSIZE = 64
        c = [(IMSIZE * 0, IMSIZE * 0, IMSIZE * 0.08),
             (IMSIZE * 0.05, IMSIZE * 0.3, IMSIZE * 0.06),
             (IMSIZE * 0.1, IMSIZE * 0.08, IMSIZE * 0.06),
             (IMSIZE * 0.1, IMSIZE * 0.03, IMSIZE * 0.03),
             (IMSIZE * 0.16, IMSIZE * 0.03, IMSIZE * 0.02)][severity - 1]

        image = np.array(image, dtype=np.float32) / 255.
        shape = image.shape
        shape_size = shape[:2]

        # random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
        dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255
