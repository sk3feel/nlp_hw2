# BART106M

`solution.ipynb`: обучение и локальный инференс. Код: `data.py`, `model.py`, `train.py`, `inference.py`. Зависимости: `requirements.txt`.

## Данные и обучение

Источник: `SlavaYus/tinysms_qa_12m`, revision `b2724689ff079e9c22efd548fe134bfb29036dbb`. Исходные60 parquet: `data/raw_dataset/data/train/part-00000.parquet` ... `part-00059.parquet`.

`data/prepared_data.zip` содержит подготовленную выборку того же датасета: 5820047 пар после фильтров, исключения holdout и токенизации. Это необязательный готовый кэш. Код может пересобрать его из исходных parquet; tokenizer и индексы выборки включены в репозиторий.

Для обучения с подготовкой из исходных данных в первой ячейке:

```python
INFERENCE_ONLY = False
PREPARE_FROM_RAW = True
RUN_TRAINING = True
USE_READY_WEIGHTS = False
```

Каталог `PREPARED` должен быть новым. Для использования готового кэша вместо пересборки: `PREPARE_FROM_RAW = False`.

## Инференс

По умолчанию notebook запускает инференс. Он требует отдельных обученных весов в `weights/avg3_mid/`; веса и архивы с ними не хранятся в текущей версии репозитория.

Colab notebook: `inference_on_colab/Colab_inference_T4.ipynb`. Его входной ZIP из локальной копии проекта должен находиться на Drive: `Мой диск/nlp_hw2/inference_avg3_mid.zip`. Выберите T4 и выполните все ячейки.
