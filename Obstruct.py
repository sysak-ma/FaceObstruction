import functools
import numpy as np
import path
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageColor


MAX_COLOR = 255


def visualizePlt(imges, figsize=(10, 10), gridsize=None):
    """Visualize image or list of images via pyplot.
    If image is singular, gridsize has to be None, otherwise
    it is required to contain amount of subplots 
    greater or equal than the number of pictures.
    """
    if gridsize is None:
        plt.figure(figsize=figsize)
        plt.imshow(imges)
        plt.xticks([])
        plt.yticks([])
        return
    plt.figure(figsize=figsize)
    for i, img in enumerate(imges):
        plt.subplot(*gridsize, i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])


def getRandomImageList(lst):
    """Return random PIL Image from
    list of paths
    """
    return Image.open(np.random.choice(lst))


def getRandomImageDirectory(directory):
    """Return random PIL Image from directory,
    assuming it consists only of pictures
    """
    image_paths = path.glob.glob(directory + '/*')
    return getRandomImageList(image_paths)


def obstructShape(img, shape='rectangle', pos=(0.25, 0.25), size=(0.5, 0.5), color='gray', alpha=MAX_COLOR, **shape_kwargs):
    """Obstruct image with shape, 
    bounding box upperleft corner at pos, 
    with given size, color and alpha.
    """
    if type(img) is np.ndarray:
        img = Image.fromarray(img)
    img = img.convert('RGBA')
    
    if color == 'random':
        color = tuple(np.random.randint(0, MAX_COLOR+1, 3))
    elif type(color) is str:
        color = ImageColor.getrgb(color)
    else:
        color = tuple(min(MAX_COLOR, x if type(x) is int else int(MAX_COLOR * x))\
                      for x in color)
    
    if alpha == 'random':
        alpha = np.random.randint(0, MAX_COLOR+1)
    elif type(alpha) is float or type(alpha) is np.float64:
        alpha = min(int(MAX_COLOR * alpha), MAX_COLOR)
    else:
        alpha = min(MAX_COLOR, int(alpha))
    
    colortup = color + (alpha,)
    
    if size == 'random':
        size = (np.random.randint(0, img.size[0]),
                np.random.randint(0, img.size[1]))
    else:
        size = tuple(
            min(img.size[i], size[i] if type(size[i]) is int else int(size[i] * img.size[i]))\
            for i in range(2)
        )
        
    if pos == 'random':
        pos = (np.random.randint(0, img.size[0] + 1 - size[0]),
               np.random.randint(0, img.size[1] + 1 - size[1]))
    else:
        pos = tuple(
            min(img.size[i] + 1 - size[i], pos[i] if type(pos[i]) is int else int(pos[i] * img.size[i]))\
            for i in range(2)
        )
        
    x1 = pos[0]
    y1 = pos[1]
    
    x2 = x1 + size[0]
    y2 = y1 + size[1]
    
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    getattr(draw, shape)(((x1, y1), (x2, y2)), fill=colortup, **shape_kwargs)
    
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    return img


@functools.lru_cache()
def openTemplate(templ_path):
    return Image.open(templ_path)


def obstructEyesSimple(img, color='gray', alpha=MAX_COLOR):
    """Obstruct eyes region with rectangle, assuming the picture
    only consists of the face, is oriented properly and is close to square.
    """
    return obstructShape(img, pos=(1/8, 1/8), size=(3/4, 1/3), color=color, alpha=alpha)


def obstructEyesSunglasses(img, templ_path="./Templates/Sunglasses.png"):
    """Obstruct eyes with drawn sunglasses, assuming the picture
    only consists of the face, is oriented properly and is close to square.
    """
    if type(img) is np.ndarray:
        img = Image.fromarray(img)
    glasses = openTemplate(templ_path).convert('RGBA').resize(img.size)
    return Image.alpha_composite(img.convert('RGBA'), glasses)


def obstructEyesEyepatch(img, templ_path="./Templates/Eyepatch.png"):
    """Obstruct eyes with drawn eyepatch, assuming the picture
    only consists of the face, is oriented properly and is close to square.
    """
    if type(img) is np.ndarray:
        img = Image.fromarray(img)
    glasses = openTemplate(templ_path).convert('RGBA').resize(img.size)
    return Image.alpha_composite(img.convert('RGBA'), glasses)
    
    
def obstructMouthSimple(img, color='gray', alpha=MAX_COLOR):
    """Obstruct mouth region with rectangle, assuming the picture
    only consists of the face, is oriented properly and is close to square.
    """
    return obstructShape(img, pos=(3/16, 2/3-1/16), size=(5/8, 1/3), color=color, alpha=alpha)


def obstructMouthBlackMask(img, templ_path="./Templates/BlackMask.png"):
    """Obstruct mouth with drawn black mask, assuming the picture
    only consists of the face, is oriented properly and is close to square.
    """
    if type(img) is np.ndarray:
        img = Image.fromarray(img)
    mask = openTemplate(templ_path).convert('RGBA').resize(img.size)
    return Image.alpha_composite(img.convert('RGBA'), mask)


def obstructMouthWhiteMask(img, templ_path="./Templates/WhiteMask.png"):
    """Obstruct mouth with drawn white mask, assuming the picture
    only consists of the face, is oriented properly and is close to square.
    """
    if type(img) is np.ndarray:
        img = Image.fromarray(img)
    mask = openTemplate(templ_path).convert('RGBA').resize(img.size)
    return Image.alpha_composite(img.convert('RGBA'), mask)


def obstructMouthScarf(img, templ_path="./Templates/Scarf.png"):
    """Obstruct mouth with drawn scarf, assuming the picture
    only consists of the face, is oriented properly and is close to square.
    """
    if type(img) is np.ndarray:
        img = Image.fromarray(img)
    mask = openTemplate(templ_path).convert('RGBA').resize(img.size)
    return Image.alpha_composite(img.convert('RGBA'), mask)


def obstructNoseSimple(img, color='gray', alpha=MAX_COLOR):
    """Obstruct nose region with rectangle, assuming the picture
    only consists of the face, is oriented properly and is close to square.
    """
    return obstructShape(img, pos=(0.3, 0.3), size=(0.4, 0.4), color=color, alpha=alpha)


def obstructHeadHood(img, templ_path="./Templates/Hood.png"):
    """Obstruct head with drawn hood, assuming the picture
    only consists of the face, is oriented properly and is close to square.
    """
    if type(img) is np.ndarray:
        img = Image.fromarray(img)
    hood = openTemplate(templ_path).convert('RGBA').resize(img.size)
    return Image.alpha_composite(img.convert('RGBA'), hood)