import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class UncertaintyMLP(nn.Module):
    """
    Легковесный MLP Head для предсказания динамики обучения из фичей.
    Архитектура: 512 -> 256 (ReLU, Dropout) -> 64 (ReLU) -> 2 (AUM, Jitter)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(description="Train MLP to predict AUM and Jitter")
    parser.add_argument('--smoke_test', action='store_true', help='Use only 200 samples')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    # Устройство (Apple Silicon приоритет)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Используемое устройство: {device}")

    # Загрузка данных
    metrics_path = 'outputs/metrics.csv'
    embeddings_path = 'outputs/embeddings.pt'

    if not os.path.exists(metrics_path) or not os.path.exists(embeddings_path):
        print("Ошибка: Отсутствуют файлы данных. Сначала запустите compute_dynamics.py и extract_embeddings.py")
        return

    print("Загрузка метрик и эмбеддингов...")
    metrics_df = pd.read_csv(metrics_path).dropna(subset=['aum', 'jitter'])

    # weights_only=False необходим для загрузки словарей в новых версиях PyTorch
    pt_data = torch.load(embeddings_path, weights_only=False)
    pt_sample_ids = pt_data['sample_ids'].numpy()
    pt_embeddings = pt_data['embeddings']

    if args.smoke_test:
        print("ВНИМАНИЕ: Запуск в режиме smoke_test (200 примеров).")
        args.epochs = 2  # Уменьшаем количество эпох для быстрого теста
        metrics_df = metrics_df.head(200)

    # Выравнивание эмбеддингов и метрик по sample_id
    id_to_idx = {sid: i for i, sid in enumerate(pt_sample_ids)}

    # Оставляем только те метрики, для которых есть эмбеддинги
    metrics_df = metrics_df[metrics_df['sample_id'].isin(id_to_idx.keys())].copy()

    X = torch.stack([pt_embeddings[id_to_idx[sid]] for sid in metrics_df['sample_id']])
    Y = torch.tensor(metrics_df[['aum', 'jitter']].values, dtype=torch.float32)

    # Разбиение на Train / Val (80% / 20%)
    X_train, X_val, Y_train, Y_val = train_test_split(X.numpy(), Y.numpy(), test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Инициализация модели, лосса и оптимизатора
    model = UncertaintyMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Начало обучения MLP Head (Train: {len(X_train)}, Val: {len(X_val)})...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        print(
            f"Эпоха {epoch}/{args.epochs} | Train MSE: {train_loss / len(X_train):.4f} | Val MSE: {val_loss / len(X_val):.4f}")

    # Финальная оценка и раздельный MSE
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu()
            all_preds.append(preds)
            all_trues.append(batch_y)

    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)

    # Считаем MSE раздельно для AUM (индекс 0) и Jitter (индекс 1)
    mse_aum = nn.functional.mse_loss(all_preds[:, 0], all_trues[:, 0]).item()
    mse_jitter = nn.functional.mse_loss(all_preds[:, 1], all_trues[:, 1]).item()

    print("\n=========================================")
    print(f"Финальный Val MSE для AUM:    {mse_aum:.4f}")
    print(f"Финальный Val MSE для Jitter: {mse_jitter:.4f}")
    print("=========================================")

    # Построение графиков
    os.makedirs('plots', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AUM Scatter
    axes[0].scatter(all_trues[:, 0].numpy(), all_preds[:, 0].numpy(), alpha=0.5, color='blue')
    axes[0].plot([all_trues[:, 0].min(), all_trues[:, 0].max()],
                 [all_trues[:, 0].min(), all_trues[:, 0].max()], 'k--', lw=2)
    axes[0].set_title('AUM: True vs Predicted')
    axes[0].set_xlabel('True AUM')
    axes[0].set_ylabel('Predicted AUM')
    axes[0].grid(True)

    # Jitter Scatter
    axes[1].scatter(all_trues[:, 1].numpy(), all_preds[:, 1].numpy(), alpha=0.5, color='red')
    axes[1].plot([all_trues[:, 1].min(), all_trues[:, 1].max()],
                 [all_trues[:, 1].min(), all_trues[:, 1].max()], 'k--', lw=2)
    axes[1].set_title('Jitter: True vs Predicted')
    axes[1].set_xlabel('True Jitter')
    axes[1].set_ylabel('Predicted Jitter')
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = 'plots/mlp_predictions_accuracy.png'
    plt.savefig(plot_path)
    plt.close()

    print(f"Scatter plots сохранены в {plot_path}")


if __name__ == '__main__':
    main()