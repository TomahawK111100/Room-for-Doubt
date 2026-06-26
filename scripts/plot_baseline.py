import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score


def main():
    metrics_path = 'outputs/metrics.csv'
    traj_path = 'outputs/raw_trajectories.csv'

    if not os.path.exists(metrics_path) or not os.path.exists(traj_path):
        print(f"Ошибка: Не найдены файлы данных. Проверьте наличие {metrics_path} и {traj_path}")
        return

    print("Загрузка данных...")
    metrics = pd.read_csv(metrics_path)
    traj = pd.read_csv(traj_path)

    # Вычисляем mean_confidence из сырых траекторий и добавляем к метрикам
    mean_conf = traj.groupby('sample_id')['confidence'].mean().reset_index()
    mean_conf.rename(columns={'confidence': 'mean_confidence'}, inplace=True)
    metrics = metrics.merge(mean_conf, on='sample_id', how='left')

    os.makedirs('plots', exist_ok=True)

    # Устанавливаем красивый стиль для графиков
    sns.set_theme(style="whitegrid")

    print("Построение графика 1: Распределение AUM...")
    plt.figure(figsize=(8, 6))
    # Используем KDE и гистограмму, нормируем независимо (common_norm=False),
    # чтобы было видно форму распределения даже при малом количестве шумных меток.
    sns.histplot(
        data=metrics, x='aum', hue='is_noisy',
        bins=30, kde=True, stat="density", common_norm=False,
        palette={False: "blue", True: "red"}
    )
    plt.title('AUM Distribution: Clean vs Noisy Labels')
    plt.xlabel('Area Under the Margin (AUM)')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig('plots/aum_distribution.png')
    plt.close()

    print("Построение графика 2: Карта датасета (Cartography Analogue)...")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=metrics, x='mean_confidence', y='jitter', hue='is_noisy',
        palette={False: "blue", True: "red"}, alpha=0.7, edgecolor=None
    )
    plt.title('Dataset Cartography (Confidence vs Jitter)')
    plt.xlabel('Mean Confidence')
    plt.ylabel('Jitter')
    plt.tight_layout()
    plt.savefig('plots/dataset_cartography.png')
    plt.close()

    print("Вычисление метрики AUROC...")
    try:
        # Для ROC AUC позитивный класс = шум (is_noisy == True).
        # Шумные объекты обычно имеют низкий AUM, поэтому берем AUM со знаком минус.
        y_true = metrics['is_noisy'].astype(int)
        y_score = -metrics['aum']

        auroc = roc_auc_score(y_true, y_score)
        print(f"\n=========================================")
        print(f"AUROC для детекции шума по метрике AUM: {auroc:.4f}")
        print(f"=========================================")
    except ValueError as e:
        print("\n[Внимание] Невозможно вычислить AUROC.")
        print("Скорее всего, в выборке smoke-теста попался только один класс (например, нет шумных меток).")
        print(f"Детали ошибки: {e}")

    print("\nГрафики успешно сохранены в папку 'plots/':")
    print("- plots/aum_distribution.png")
    print("- plots/dataset_cartography.png")


if __name__ == '__main__':
    main()