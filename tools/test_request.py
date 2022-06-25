from typing import Dict
import requests
import cv2
from tqdm import tqdm

def draw_contours(image_array, metadata):
    """ Функция отрисовки контура и подсчета кол-во особей
    Обрабатываем результат работы моделей, извлекая полученный класс животного
    Args:
        image_array: arr - массив-представление изображения
        metadata: json - словарь, содержащий ответ работы моделей
    Return:
        counter_dict: dict - словарь с кол-вом определенных животных
    """

    for bbox in tqdm(metadata["bbox"]):
        class_name = bbox['class_name']

        # confidence = bbox['confidence']

        topLeftCorner = (bbox['bbox']['x1'], bbox['bbox']['y1'])
        botRightCorner = (bbox['bbox']['x2'], bbox['bbox']['y2'])

        # center_coords = int((botRightCorner[0] + topLeftCorner[0]) / 2), int((botRightCorner[1] + topLeftCorner[1]) / 2)
        cv2.rectangle(
            image_array,
            topLeftCorner,
            botRightCorner,
            (255, 0, 0), 
            1
        )
        cv2.putText(image_array, f'{class_name}', 
                            topLeftCorner,
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 0, 0),
                            1,
                            1)
    return 
# Для запуска на вашей машине нужно поменять URL на ваш локальный ip 
URL = 'http://10.10.67.145:5010/api/ekb_service'

content_type = 'image/jpeg'
headers = {'content-type': content_type}

def test_req(img_path: str) -> Dict:
    image_array = cv2.imread(img_path)
    _, img_encoded = cv2.imencode('.jpg', image_array)
    data = img_encoded.tostring()
    # import pdb;pdb.set_trace()
    response = requests.post(URL, data=data, headers=headers)
    metadata = response.json()['image']
    
    return metadata, image_array

def main():
    img_path = 'test_images/998.jpg'
    res, img = test_req(img_path)
    draw_contours(img, res)
    cv2.imwrite("test_images/test.jpg", img)
    print(res['bbox'])

if __name__=="__main__":
    main()