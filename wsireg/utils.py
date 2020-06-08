import cv2
import numpy as np


#TODO new loss function: Use Normalized Gradient Field measure instead of plain norm?
def m_norm(im1, im2):
    gray1 = ensure_gray(im1)
    gray2 = ensure_gray(im2)
    diff = gray1 - gray2
    m_norm = np.mean(abs(diff))

    return m_norm

def ensure_gray(img):
    """WARNING: img must be in RGB/A or GRAY format, not BGR/A format."""
    if len(img.shape) == 4:
        return cv2.cvtColor(img, code=cv2.COLOR_RGBA2GRAY)
    elif len(img.shape) == 3:
        return cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
    elif len(img.shape) == 2:
        return img
    else:
        print("WARNING: ensure_gray does not recognize the image format.")
        return None

def make_test_img(shape=(250, 250)):
    """Generates a test image with colors arranged so that any transformation is recognizable."""
    stripe = int(shape[0]/10)
    bar = int(shape[1]/10)
    img = np.full((*shape, 3), 0, np.uint8)

    # img[:,100:150,0] = 255
    img[:, int((shape[1]-bar)/2):int((shape[1]+bar)/2), 0] = 255
    # img[:,:20,1] = 255
    img[:,:bar,1] = 255
    img[:,-bar:,2] = 255
    img[:stripe,:,2] = 255
    img[-stripe:,:,1] = 255

    return img
