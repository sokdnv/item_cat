import streamlit as st
import torch
import torch.nn as nn
import json
from transformers import AutoTokenizer, AutoModel
import re

# Гиперпараметры модели
dropout = 0.15
lev_1 = 15
lev_2 = 130
lev_3 = 980


def clean_text(text):
    # Удаление HTML тегов
    text = re.sub(r'<.*?>', '', text)
    # Удаление эмодзи и спецсимволов
    text = re.sub(r'[^\w\s,.!?]', '', text)
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Функция для нормализации текста
def normalize_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Замена чисел на специальный токен
    text = re.sub(r'\d+', '<NUM>', text)
    return text


# Основная функция предобработки текста
def preprocess_text(text):
    text = clean_text(text)
    text = normalize_text(text)
    return text


# Определение архитектуры модели
class HierarchicalClassifier(nn.Module):
    def __init__(self):
        super(HierarchicalClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

        # Полносвязные слои для первого уровня категоризации (15 классов)
        self.fc1 = nn.Sequential(
            nn.Linear(312, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, lev_1)  # 15 классов на первом уровне
        )

        # Полносвязные слои для второго уровня категоризации (130 классов)
        self.fc2 = nn.Sequential(
            nn.Linear(312 + lev_1, 256),  # добавляем эмбеддинги + первый уровень
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, lev_2)  # 130 классов на втором уровне
        )

        # Полносвязные слои для третьего уровня категоризации (980 классов)
        self.fc3 = nn.Sequential(
            nn.Linear(312 + lev_2, 256),  # добавляем эмбеддинги + 2 уровень
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, lev_3)  # 980 классов на третьем уровне
        )

    def forward(self, input_ids, attention_mask):
        # Пропускаем вход через BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.pooler_output  # Используем CLS-токен для эмбеддинга текста

        # Первый уровень классификации
        level1_output = self.fc1(cls_embedding)

        # Второй уровень классификации
        level2_input = torch.cat((cls_embedding, level1_output), dim=1)
        level2_output = self.fc2(level2_input)

        # Третий уровень классификации
        level3_input = torch.cat((cls_embedding, level2_output), dim=1)
        level3_output = self.fc3(level3_input)

        return level1_output, level2_output, level3_output


# Загрузка токенизатора
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")

# Инициализация и загрузка весов модели
model = HierarchicalClassifier().to(DEVICE)
model.load_state_dict(torch.load("weights_v2.pth", map_location=DEVICE))

# Загрузка словарей из JSON файлов
with open('level1_categories.json', 'r', encoding='utf-8') as f:
    idx_to_category_level1 = json.load(f)

with open('level2_categories.json', 'r', encoding='utf-8') as f:
    idx_to_category_level2 = json.load(f)

with open('level3_categories.json', 'r', encoding='utf-8') as f:
    idx_to_category_level3 = json.load(f)


# Функция предсказания
def predict_category(text):
    # Токенизация текста и преобразование в тензоры

    text = preprocess_text(text)

    encoded_input = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )
    input_ids = encoded_input['input_ids'].to(DEVICE)
    attention_mask = encoded_input['attention_mask'].to(DEVICE)

    # Прогоняем через модель
    model.eval()
    with torch.no_grad():
        level1_logits, level2_logits, level3_logits = model(input_ids, attention_mask)

    # Получаем предсказания (на каждом уровне)
    level1_pred = torch.argmax(level1_logits, dim=1).item()
    level2_pred = torch.argmax(level2_logits, dim=1).item()
    level3_pred = torch.argmax(level3_logits, dim=1).item()

    # Преобразуем индексы предсказаний в категории
    level1_category = idx_to_category_level1.get(str(level1_pred), "Unknown")
    level2_category = idx_to_category_level2.get(str(level2_pred), "Unknown")
    level3_category = idx_to_category_level3.get(str(level3_pred), "Unknown")

    return {
        'Уровень 1': level1_category,
        'Уровень 2': level2_category,
        'Уровень 3': level3_category
    }


# Streamlit интерфейс
st.title("Категоризация товаров")
st.write("Введите название товара, чтобы узнать предсказанную категорию.")

text = st.text_input("Название товара", "")

if st.button("Определить категорию"):
    if text:
        predictions = predict_category(text)
        st.write("Предсказанные категории:")
        for level, category in predictions.items():
            st.write(f"{level}: {category}")
    else:
        st.write("Пожалуйста, введите название товара.")
