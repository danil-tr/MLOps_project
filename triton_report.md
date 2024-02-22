### Проведение тестов
Запуск контейнера:
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.04-py3-sdk

Внутри контейнера:
perf_analyzer -m onnx_classifier -u localhost:8500 --concurrency-range 2:2 --shape float_input:1,4

### Система:
- ОС: Ubuntu 22.04.1 LTS
- Процессор: Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz
- Кол-во ядер: 4
- Использованная RAM: 8 ГБ
- Выделенная контейнеру shared memory: 64 МБ

### Структура:
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


### Динамический батчинг
Найдем оптимальное значение max_queue_delay_microseconds. Будем менять эту величину, пока не сойдемся к оптимальному значению
Меняем  max_queue_delay_microseconds
- 0 ms:
    Throughput: 6784.91 infer/sec
    Avg latency: 294 usec (standard deviation 351 usec)
- 10 ms:
    Throughput: 5914.91 infer/sec
    Avg latency: 337 usec (standard deviation 288 usec)
- 50 ms:
    Throughput: 4690.18 infer/sec
    Avg latency: 425 usec (standard deviation 239 usec)
- 100 ms:
    Throughput: 3793.93 infer/sec
    Avg latency: 526 usec (standard deviation 256 usec)
- 200 ms:
    Throughput: 3080.18 infer/sec
    Avg latency: 648 usec (standard deviation 190 usec)
- 300 ms:
    Throughput: 1355.2 infer/sec
    Avg latency: 1473 usec (standard deviation 664 usec)
- 1000 ms:
    Throughput: 740.242 infer/sec
    Avg latency: 2698 usec (standard deviation 501 usec)

Наблюдается прямо пропорциональная зависимость ухудшения результа от времени ожидания, связанная с устройстовом, на котором проводились тесты.
Вывод: на имеющемся устройстве динамический батчинг не помогает ускорить ни одну из метрик.

### Найдем оптимальное кол-во инстансов модели.
Меняем кол-во инстансов модели при фиксированном max_queue_delay_microseconds = 50 (близкое значение к оптимальному)

- 1:
    Throughput: 4690.18 infer/sec
    Avg latency: 425 usec (standard deviation 239 usec)
- 2:
    Throughput: 3951.403 infer/sec
    Avg latency: 529 usec (standard deviation 334 usec)


Не наблюдается улучшения метрик при увеличении кол-ва инстансов модели
Вывод: оставим 1 инстанс модели
