# BART106M

## Обучение

Скачайте train-данные [SlavaYus/tinysms_qa_12m](https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m). Из папки проекта выполните:

```bash
pip install -r requirements.txt
python prepare_data.py --raw-dir /путь/к/data/train
```

Скрипт создаст `data/prepared_data.zip`. Откройте `solution.ipynb` и выполните все ячейки для обучения, усреднения весов и инференса.

## Colab-инференс

Загрузите `inference_avg3_mid.zip` на Drive: `Мой диск/nlp_hw2/inference_avg3_mid.zip`.
Откройте `inference_on_colab/Colab_inference_T4.ipynb`, выберите T4 и выполните все ячейки.
