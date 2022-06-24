from typing import Dict
import requests
import cv2

# Для запуска на вашей машине нужно поменять URL на ваш локальный ip 
URL = 'http://0.0.0.0:5010/api/habarovsk_service'

content_type = 'image/jpeg'
headers = {'content-type': content_type}

def test_req(img_path: str) -> Dict:
    image_array = cv2.imread(img_path)
    _, img_encoded = cv2.imencode('.jpg', image_array)
    data = img_encoded.tostring()
    # import pdb;pdb.set_trace()
    response = requests.post(URL, data=data, headers=headers)
    metadata = response.json()['image']
    
    return metadata

def main():
    img_path = 'test_images/1.jpg'
    res = test_req(img_path)
    print(res['bbox'])

if __name__=="__main__":
    main()