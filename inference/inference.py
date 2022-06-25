from typing import Dict
import requests
import cv2
from tqdm import tqdm
import argparse
import os

from utils import draw_contours

parser = argparse.ArgumentParser(description='image recognition')

parser.add_argument("--image", dest='img_path', help="image to recognition", default='')
parser.add_argument("--image_path", dest='path2images', help="path with images to recognize", default='')

arguments = parser.parse_args()

recognitions_dir = './recognition_result'

if not os.path.exists(recognitions_dir):
    os.mkdir(recognitions_dir)

# Для запуска на вашей машине нужно поменять URL на ваш локальный ip 
URL = 'http://10.10.67.145:5010/api/ekb_service'

content_type = 'image/jpeg'
headers = {'content-type': content_type}

def request2back(img_path: str) -> Dict:
    image_array = cv2.imread(img_path)
    _, img_encoded = cv2.imencode('.jpg', image_array)
    data = img_encoded.tostring()

    response = requests.post(URL, data=data, headers=headers)
    metadata = response.json()['image']
    
    return metadata, image_array


def main():

    img_path = arguments.img_path
    path2images = arguments.path2images

    if arguments.img_path:
        print('Идет распознавание одного изображения')
        res, img = request2back(img_path)
        draw_contours(img, res)

        img_name = os.path.basename(img_path)
        cv2.imwrite(f"{recognitions_dir}/{img_name}", img)

    elif path2images:
        print('Идет распознавание изображений из указанной директории')
        image_files = [i for i in os.listdir(path2images) if i.endswith('.jpg')]
        for image in tqdm(image_files):
            res, img = request2back(f'{path2images}/{image}')
            draw_contours(img, res)

            cv2.imwrite(f"{recognitions_dir}/{image}", img)
    else:
        print('Вы не передали ни одного значения')
            

if __name__=="__main__":
    main()