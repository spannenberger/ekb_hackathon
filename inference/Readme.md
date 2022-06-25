# Inference service 
#### Инструкция для получения распознанных изображений через запуск скрипта в виртуальном окружении

## ! запускать из корня проекта ! сервис распознавания должен быть поднят !
### Установка зависимостей в виртуальное окружение

```
python3 -m pip install virtualenv

python3 -m virtualenv venv

source venv/bin/activate

pip install -r inference/inference_requirements.txt
```

### Для распознавания одного изображения
```
python inference/inference.py --image "path_to_photo/example.jpg"
```
### Для распознавания всех изображений, находящихся в директории
```
python inference/inference.py --image_path "path_to_photos"
```