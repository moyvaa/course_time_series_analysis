#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Курсовой проект: Первичный анализ многомерного временного ряда
Тема: Прогнозирование концентрации PM2.5 в Пекине
Набор данных: Beijing PM2.5 Data Set
Источник: UCI Machine Learning Repository
Задача: продемонстрировать этапы 1-7 первичного анализа.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

DATA_PATH = 'data/PRSA_data_2010.1.1-2014.12.31.csv'

if not os.path.exists(DATA_PATH):
    import requests
    from io import StringIO
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv'
    print("Скачиваем данные...")
    resp = requests.get(url)
    resp.encoding = 'utf-8'
    df = pd.read_csv(StringIO(resp.text))
    os.makedirs('data', exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

print("Первые 5 строк:\n", df.head())
print("\nРазмерность:", df.shape)
print("\nИнформация о типах данных:")
df.info()

df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df.set_index('datetime', inplace=True)
df.drop(['No', 'year', 'month', 'day', 'hour'], axis=1, inplace=True)
df.columns = ['PM2.5', 'Точка_росы', 'Температура', 'Давление',
              'Направление_ветра', 'Скорость_ветра', 'Осадки_час', 'Осадки_сутки']

print("После обработки:\n", df.head())
print("Длина ряда:", len(df))

print("\n=== ВЫВОДЫ ЭТАПА 1 ===")
print("Данные успешно загружены. Многомерный ряд: 8 каналов, 43824 записи.")
print("Временная метка – каждый час с 01.01.2010 по 31.12.2014.")
print("Типы данных: числовые float64, кроме 'Направление_ветра' (object). Пропуски — NaN.")

#2
target = 'PM2.5'

fig, axes = plt.subplots(len(df.columns), 1, figsize=(16, 12), sharex=True)
for i, col in enumerate(df.columns):
    ax = axes[i]
    ax.plot(df.index, df[col], linewidth=0.5, alpha=0.8)
    ax.set_ylabel(col, fontsize=9)
    ax.grid(True)
axes[0].set_title('Все каналы временного ряда', fontsize=14)
axes[-1].set_xlabel('Дата')
plt.tight_layout()
plt.savefig('plots/raw_all_channels.png', dpi=150)
plt.show()

train_end = '2013-12-31 23:00:00'
plt.figure(figsize=(14, 5))
plt.plot(df.index, df[target], linewidth=0.6)
plt.axvline(pd.Timestamp(train_end), color='red', linestyle='--', label='Train / Test')
plt.title('PM2.5 и граница обучающей/тестовой выборок')
plt.legend()
plt.tight_layout()
plt.savefig('plots/target_with_split.png', dpi=150)
plt.show()

print("\nЭТАП 2:")
print("1. Тренд долгосрочный отсутствует, видна годовая сезонность (зимние пики).")
print("2. Повторяющиеся паттерны: суточные и годовые колебания.")
print("3. Ряд зашумлён, но периодичности предсказуемы.")
print("4. Пропуски видны как разрывы в линиях, особенно в начале.")

#3
desc = df.describe().T
desc['IQR'] = desc['75%'] - desc['25%']
print("\nОписательные статистики:\n", desc)

# Частота
time_deltas = df.index.to_series().diff().dropna()
freq = time_deltas.value_counts().idxmax()
print(f"Преобладающий интервал: {freq}")

print("\nЭТАП 3:")
print("Распределение PM2.5 асимметрично (среднее > медиана).")
print("Интервалы равномерны (1 час). Разброс по каналам сильно разный — потребуется масштабирование.")

#4
missing = (df.isnull().sum() / len(df)) * 100
print("Доля пропусков (%):\n", missing)

print("\nКоличество выбросов (правило 3σ):")
for col in df.select_dtypes('number').columns:
    mean, std = df[col].mean(), df[col].std()
    n_outliers = (np.abs(df[col] - mean) > 3 * std).sum()
    print(f"{col}: {n_outliers}")

num_cols = df.select_dtypes('number').columns
fig, axes = plt.subplots(1, len(num_cols), figsize=(18, 5))
for ax, col in zip(axes, num_cols):
    df.boxplot(column=col, ax=ax)
    ax.set_title(col, fontsize=8)
plt.suptitle('Диаграммы размаха')
plt.tight_layout()
plt.savefig('plots/boxplots.png')
plt.show()

print("4 ЭТАП:")
print("Пропуски есть во всех каналах (до 4%). Выбросы многочисленны в PM2.5, ветре, осадках.")

#5
scaler = StandardScaler()
scaled = scaler.fit_transform(df[num_cols].dropna())
df_scaled = pd.DataFrame(scaled, columns=num_cols)
plt.figure(figsize=(12, 5))
df_scaled.boxplot(rot=45)
plt.title('Сравнение масштабов (стандартизированные)')
plt.tight_layout()
plt.savefig('plots/range_comparison.png')
plt.show()

print("ЭТАП 5:")
print("Диапазоны несоизмеримы. Требуется нормализация/стандартизация.")

#6
corr = df[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Матрица корреляций')
plt.tight_layout()
plt.savefig('plots/corr_heatmap.png')
plt.show()

print("Корреляции с PM2.5:\n", corr['PM2.5'].drop('PM2.5').sort_values(ascending=False))

print("ЭТАП 6:")
print("Температура и точка росы сильно коррелируют (>0.8). Мультиколлинеарность.")

# 7
ts = df[target].loc['2013-01-01':'2013-12-31'].asfreq('h')
ts.interpolate(method='linear', inplace=True)

decomp = seasonal_decompose(ts, model='additive', period=24)
trend = decomp.trend
seasonal = decomp.seasonal
resid = decomp.resid

fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
axes[0].plot(ts, linewidth=0.8); axes[0].set_ylabel('Исходный')
axes[1].plot(trend); axes[1].set_ylabel('Тренд')
axes[2].plot(seasonal); axes[2].set_ylabel('Сезонность')
axes[3].plot(resid); axes[3].set_ylabel('Остатки')
plt.suptitle('Декомпозиция PM2.5 (аддитивная, period=24)')
plt.tight_layout()
plt.savefig('plots/decomposition.png')
plt.show()

signal = trend + seasonal
var_s = np.var(signal.dropna())
var_n = np.var(resid.dropna())
snr = 10 * np.log10(var_s / var_n)
print(f"SNR = {snr:.2f} дБ")

plt.figure(figsize=(10, 5))
sns.histplot(resid.dropna(), bins=60, kde=True, color='purple')
plt.title('Распределение остатков')
plt.tight_layout()
plt.savefig('plots/residual_hist.png')
plt.show()

print("ЭТАП 7:")
print("Слабый тренд, яркая суточная сезонность.")
print(f"SNR около {snr:.1f} дБ – качество хорошее.")
print("Распределение остатков почти нормальное, но с тяжёлыми хвостами.")

print("\nАнализ завершён. Графики сохранены в папку plots.")


