
#### Необходима установка (версия указана для примера)
Сonda (version 24.1.1)<br>
Poetry (version 1.7.1)<br>
Docker (version 25.0.2)<br>
При установке, если не указано иное, действие в терминале происходят из корневой папки проекта.<br><br>

#### Предварительные действия:
1. Сделать ```git clone https://github.com/danil-tr/MLOps_project.git```<br><br>
2. Создать виртаульное окружение для проекта. Как вариант создать его с помощью conda в папке с проектом:
```conda create --prefix {путь к проекту}/.venv python=3.11```<br><br>
3. Активировать его: ```source activate {путь к проекту}/.venv```<br><br>

4. Далее установить зависимости в созданный .venv используя poetry<br><br>
    4.1. _(Опционально)_ Настройка poetry для использования виртуального окружения внутри проекта: ```poetry config virtualenvs.in-project true```<br><br>
	4.2. ```poetry install``` (Проверить, что все зависимости имеются можно с помощью ```conda list```, при ошибке попробовать ```poetry lock```)<br><br>
	4.3.  _(Опционально)_ Сделать build: ```poetry build```<br><br>
	4.4.  _(Опционально)_ Проверить с помощью git hooks: ```pre-commit run -a```<br><br>

#### Поднятие сервера mlflow (необходимо перед запуском train.py)<br>
1. Для логирования метрик необходимо передать адрес сервера mlflow в /config/config.yaml (по умолчанию указан адрес http://localhost:5001/)<br><br>
2. Чтобы поднять свой сервер на адресе http://localhost:5001/, можно:<br><br>
	2.1. Использовать docker-compose.yml, хранящийся в папке tracking _server. В нем создается сервис mlflow с postgresql и minio s3 для хранения артефактов.<br>
Для этого внутри папки tracking_server необходимо прописать```sudo docker compose up -d --build```<br>
Далее зайти в minio s3 http://127.0.0.0:9001/. Ввести MINIO_ACCESS_KEY
MINIO_SECRET_ACCESS_KEY из tracking_server/.env файла, создав новый ключ. Далее перезапустить контейнер. При этом создастся bucket для хранения артефактов.<br><br>
	2.2.  _(Опционально)_ Для дальнейших запусков, когда создан bucket, можно использовать run_server.py. Из директории tracking_server прописать ```python3 run_server.py```<br><br>
	2.3. _(Обязательно для инференса с mlflow)_ В папке iris_classifier запуcтить ```python3 train.py```. При этом по адресу http://localhost:5001/ можно будет посмотреть логи эксперимента. Также модель сохранится в формате onnx и для инференса с помощью mlflow в папке model_result.<br><br>
	2.4. _(Опционально)_ В папке iris_classifier запуcтить ```python3 infer.py```. При этом в папке model_result появится файл prediction.csv с запуском модели на тестовом датасете.<br><br>

#### Инференс mlflow
1. Для инференса модели c помощью mlflow необходимо указать адрес сервера в /config/config.yaml (по умолчанию указан адрес http://localhost:5003/)<br><br>
2. Чтобы поднять свой сервер для инференса, нужно:<br><br>
2.1. В папке iris_classifier запутить (предварительно подняв mlflow tracking server) ```python3 train.py```<br><br>
2.2. Запустить (в отдельном терминале, который будет запущен, пока необходим локальный сервер для инференса)<br> ```mlflow models serve -m ./model_result/mlflow_model --env-manager local --host 127.0.0.1 --port 5003```<br><br>
2.3. В папке iris_classifier запустить ```python3 run_mlflow_model.py```. При этом на созданный сервер отправится тестовый запрос и в консоль выведется ответ.<br><br>

#### Triton
Структура:<br>
```
triton_inference_server
├── client.py
├── docker-compose.yaml
├── Dockerfile
└── model_repository
    └── onnx_classifier
        ├── 1
        │   ├── convert_model.py
        │   └── model.onnx.dvc
        └── config.pbtxt
```
1) Markdown отчет с оптимизацией хранится в корневой директории - triton_report.md<br><br>
2) Необходимо сформировать файл model.onnx в папке triton_inference_server/model_repository/onnx_classifier/1:<br><br>
2.1. Сделать ```dvc pull```. Тогда model.onnx загрузится из dvc<br><br>
2.2. В папке triton_inference_server запустить ```sudo docker compose up```<br><br>
2.3. В папке triton_inference_server запустить  ```python3 client.py```. При этом серверу отправится тестовый батч, в терминал выведется ответ.<br><br>
