from PIL import Image, ImageEnhance
import numpy as np
import cv2


def read_image_request(input_image):
    """Функция для обрабтки изображения
    
    Args:
        request_from_bot: bytearray - байтовое представление изображения
        
    Return:
        img: np.array - numpy массив-представление изображения
    """

    nparr = np.fromstring(input_image.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img

def image_augmentations(img: np.array):
    """Добавление контраста изображению
    
    Args:
        img: np.array - исходное изображение
        
    Return:
        img: np.array - аугментированное изображение
    """

    enhancer = ImageEnhance.Contrast(Image.fromarray(img))
    img = np.array(enhancer.enhance(1.5))

    return img