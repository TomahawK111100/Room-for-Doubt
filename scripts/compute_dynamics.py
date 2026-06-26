import os
import sys
import pandas as pd

# Добавляем корень проекта в sys.path для импорта датасета
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.room_for_doubt.data.dataset import CIFAR10N


def main():
    traj_path = 'outputs/raw_trajectories.csv'

    if not os.path.exists(traj_path):
        print(f"Ошибка: файл {traj_path} не найден. Сначала запустите train_classifier.py")
        return

    print("Загрузка сырых траекторий...")
    df = pd.read_csv(traj_path)

    print("Подтягивание чистых и шумных меток из CIFAR-10N...")
    # download=False, так как мы уже всё скачали на предыдущих шагах
    dataset = CIFAR10N(root='./data', noise_type='aggregate', train=True, download=False)

    # Собираем метки в DataFrame для быстрого джойна.
    # Длина dataset всегда 50000, даже если мы прогнали smoke_test только на 200 примерах.
    labels_df = pd.DataFrame({
        'sample_id': range(len(dataset)),
        'clean_label': dataset.clean_labels,
        'noisy_label': dataset.noisy_labels
    })

    # Флаг зашумленности метки (строго для оценки)
    labels_df['is_noisy'] = labels_df['clean_label'] != labels_df['noisy_label']

    print("Вычисление метрик динамики: AUM, Forgetting, Jitter...")
    # Объединяем траектории с метками
    df = df.merge(labels_df, on='sample_id', how='left')

    # Сортируем по ID и эпохам, чтобы хронология была правильной
    df = df.sort_values(by=['sample_id', 'epoch'])

    # Правильность предсказания (относительно ШУМНОЙ метки, на которую мы учились)
    df['is_correct'] = df['predicted_label'] == df['noisy_label']

    # Смещаем столбцы вверх (-1), чтобы заглянуть в "следующую эпоху" для каждого примера
    df['correct_next'] = df.groupby('sample_id')['is_correct'].shift(-1)
    df['pred_next'] = df.groupby('sample_id')['predicted_label'].shift(-1)

    # Forgetting: сейчас правильно (True), а на следующей эпохе - ошибка (False)
    df['is_forgotten'] = (df['is_correct'] == True) & (df['correct_next'] == False)

    # Jitter: предсказанный класс изменился между t и t+1 (и t+1 существует)
    df['is_jitter'] = (df['predicted_label'] != df['pred_next']) & df['pred_next'].notna()

    # Агрегация метрик по каждому уникальному sample_id
    metrics = df.groupby('sample_id').agg(
        aum=('margin', 'mean'),
        forgetting_count=('is_forgotten', 'sum'),
        jitter=('is_jitter', 'sum')
    ).reset_index()

    # Присоединяем статичные метки обратно к итоговой таблице
    metrics = metrics.merge(
        labels_df[['sample_id', 'clean_label', 'noisy_label', 'is_noisy']],
        on='sample_id',
        how='left'
    )

    out_path = 'outputs/metrics.csv'
    metrics.to_csv(out_path, index=False)

    # Небольшой вывод для самопроверки
    print(f"\nГотово! Метрики сохранены в {out_path}.")
    print(f"Обработано примеров: {len(metrics)}")
    print("\nПревью данных:")
    print(metrics.head())


if __name__ == '__main__':
    main()