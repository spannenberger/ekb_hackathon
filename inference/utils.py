from tqdm import tqdm
import cv2


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

        topLeftCorner = (bbox['bbox']['x1'], bbox['bbox']['y1'])
        botRightCorner = (bbox['bbox']['x2'], bbox['bbox']['y2'])

        cv2.rectangle(image_array, topLeftCorner, botRightCorner, (255, 0, 0), 1)

        cv2.putText(image_array, f'{class_name}', topLeftCorner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, 1)