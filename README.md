# Caries Decetion Hackathon Solution
#### Данный репозиторий является решением "Кейс №3: Технологии ИИ в детской стоматологии" от команды "hyper popis [Napoleon IT]"

### Содержание
- [user guide](#user-guide):
    * [Структура репозитория](#структура-репозитория)
    * [Инструкция по использованию репозитория в docker](#docker-run)
- [Объяснение решения](#объяснение-решения)


# User guide
### Структура репозитория
- [teeth_metric_learning.ipynb](teeth_metric_learning.ipynb) - код для обучения классификатора
- [teeth-caries.ipynb](teeth-caries.ipynb) - код для обучения детектора
- [app](./app/) - папка содержащая реализацию бэка, всю логику работы моделей, инициализацию моделей
- [dash_app](./dash_app/) - папка содержащая реализацию web-интерфейса
- [backend.py](backend.py) - вспомогательный файл для запуска сервиса
- [docker-compose.yml](docker-compose.yml) - конфиг докера для сборки и поднятия сервиса и бота с нужными портами и тд
- [Dockerfile](Dockerfile) - докер файл сервиса, отвечающий за окружение и установку нужных пакетов, библиотек
- [full_base_file.csv](full_base_file.csv) - csv таблица с усредненными эмбеддингами классов - нужна для подсчета близжайших классов для изображения
- [install_models.sh](install_models.sh) - скрипт для выгрузки моделей с облака
- [INFERENCE.md](INFERENCE.md) - документация по прогону ваших данных через модель
- [inference](inference) - директория с документацией для инференса моделей(прогон изображений через модели и составление csv таблицы с результатами работы)
- [tools](tools) - директория со вспомогательными функциями
- [requirements.txt](requirements.txt) - файл со всеми необходимыми библиотеками для работы сервиса
### Пример .env file

```
DETECTION_MODEL=/workspace/source/detection_model/latest.pth
DETECTION_CONFIG=/workspace/source/detection_model/config.py
METRIC_EXTRACTOR=google/vit-base-patch16-384
METRIC_MODEL=/workspace/source/model/
METRIC_CSV_PATH=/workspace/source/model/embs.csv
DB_URL=mysql+mysqlconnector://root:example@YOUR_LOCAL_IP:3306/test_db
SERVICE_URL=YOUR_LOCAL_IP
```
#### Вместо YOUR_LOCAL_IP поставить Ваш локальный ip адрес

### Docker run
Для того чтобы поднять сервис на локальной/удаленной машине нужно:
- убедиться, что указанные порты в ```docker-compose.yml``` доступны на вашей машине
- запустить скрипт сборки docker контейнеров:
```
docker-compose build
```
- запустить скрипт поднятия сервисов:
```
docker-compose up -d
```
- Поздравляем, сервисы подняты
### Объяснение решения 
Для решения задачи мы использовали визуальные трансформеры. В сервисе используются 2 обученные модели:
1. Каскадная rcnn для детекции животных
2. Визуальный трансформер (metric learning) для классификации зубов, найденных на предыдущем шаге

