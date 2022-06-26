# Caries Detecion Hackathon Solution
#### Данный репозиторий является решением "Кейс №3: Технологии ИИ в детской стоматологии" от команды "hyper popis [Napoleon IT]"

### Содержание
- [user guide](#user-guide):
    * [Структура репозитория](#структура-репозитория)
    * [Инструкция по использованию репозитория в docker](#запуск-решения)
- [Объяснение решения](#объяснение-решения)


# User guide
### Структура репозитория
- [teeth_metric_learning.ipynb](teeth_metric_learning.ipynb) - код для обучения классификатора
- [teeth-caries.ipynb](teeth-caries.ipynb) - код для обучения детектора
- [app](./app/) - папка содержащая реализацию бэка, всю логику работы моделей, инициализацию моделей
- [dash_app](./dash_app/) - папка содержащая реализацию web-интерфейса
- [backend.py](backend.py) - вспомогательный файл для запуска сервиса
- [docker-compose.yml](docker-compose.yml) - конфиг докера для сборки и поднятия наших сервисов
- [Dockerfile](Dockerfile) - докер файл сервиса, отвечающий за окружение и установку нужных пакетов, библиотек
- [full_base_file.csv](full_base_file.csv) - csv таблица с усредненными эмбеддингами классов - нужна для подсчета близжайших классов для изображения
- [install_models.sh](install_models.sh) - скрипт для выгрузки моделей с облака
- [init_db_table.sql](init_db_table.sql) - скрипт инициализации БД и таблицы внутри нее, используемые в сервисе
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
DB_URL=mysql+mysqlconnector://root:example@YOUR_LOCAL_IP:3306/ekb_service
SERVICE_URL=YOUR_LOCAL_IP/api/ekb_service
```
#### Вместо YOUR_LOCAL_IP поставить Ваш локальный ip адрес

### Запуск решения
Для того чтобы поднять сервис на локальной/удаленной машине нужно:
- убедиться, что указанные порты в ```docker-compose.yml``` доступны на вашей машине
- запустить команду сборки:
```
docker-compose build
```
- запустить команду поднятия сервисов:
```
docker-compose up -d
```
- Поздравляем, сервисы подняты
- Далее необходимо проинициализировать базу данных (следуйте [инструкции](database/DB_README.md))
### Объяснение решения 
Для решения задачи мы использовали визуальные трансформеры. В сервисе используются 2 обученные модели:
1. Каскадная rcnn для детекции зубов
2. Визуальный трансформер (metric learning) для классификации заболеваний на зубах, найденных на предыдущем шаге

