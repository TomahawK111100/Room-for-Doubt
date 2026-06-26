import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
import sys

# Добавляем корень проекта в sys.path, чтобы импортировать dataset.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.room_for_doubt.data.dataset import CIFAR10N


class IndexedDataset(Dataset):
    """Обертка для получения индекса (sample_id) вместе с данными."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, clean_label, noisy_label = self.base_dataset[idx]
        return idx, img, clean_label, noisy_label


def calculate_margin(logits, target_labels):
    """Вычисляет margin: логит целевого класса минус максимальный логит из остальных."""
    batch_size = logits.size(0)
    target_logits = logits[torch.arange(batch_size), target_labels]

    # Клонируем логиты и маскируем целевой класс как -inf, чтобы найти второй максимум
    logits_clone = logits.clone()
    logits_clone[torch.arange(batch_size), target_labels] = -float('inf')
    max_other_logits, _ = torch.max(logits_clone, dim=1)

    margin = target_logits - max_other_logits
    return margin


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 and log trajectories")
    parser.add_argument('--smoke_test', action='store_true', help='Run 1 epoch on 200 samples on CPU')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    args = parser.parse_args()

    # Настройки для smoke_test
    if args.smoke_test:
        print("ВНИМАНИЕ: Запуск в режиме smoke_test (1 эпоха, 200 примеров, CPU).")
        args.epochs = 1
        args.batch_size = 32
        device = torch.device('cpu')
    else:
        # Добавлена поддержка Apple Metal (MPS)
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    print(f"Используемое устройство: {device}")

    # Трансформации (базовые для CIFAR-10)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Загрузка данных (добавляем флаг download=True)
    base_train_dataset = CIFAR10N(root='./data', noise_type='aggregate', train=True, transform=transform_train,
                                  download=True)
    test_dataset = CIFAR10N(root='./data', noise_type='aggregate', train=False, transform=transform_test, download=True)
    train_dataset = IndexedDataset(base_train_dataset)

    if args.smoke_test:
        train_dataset = Subset(train_dataset, range(200))
        test_dataset = Subset(test_dataset, range(100))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Инициализация модели ResNet-18 (меняем последний слой на 10 классов)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # Оптимизатор и функция потерь (reduction='none' для логирования каждого примера)
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Хранилище для траекторий
    trajectories = []

    # Для графиков
    train_losses = []
    val_losses = []

    print("Начало обучения...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_train_loss = 0.0

        for sample_ids, inputs, _, noisy_labels in train_loader:
            inputs, noisy_labels = inputs.to(device), noisy_labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            # Loss вычисляется только по шумным меткам! Чистые метки (_) игнорируются.
            per_sample_loss = criterion_none(logits, noisy_labels)
            loss = per_sample_loss.mean()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)

            # --- ЛОГИРОВАНИЕ ТРАЕКТОРИЙ ---
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                confidences, predicted_labels = torch.max(probs, dim=1)
                margins = calculate_margin(logits, noisy_labels)

                for i in range(inputs.size(0)):
                    trajectories.append({
                        'sample_id': sample_ids[i].item(),
                        'epoch': epoch,
                        'loss': per_sample_loss[i].item(),
                        'confidence': confidences[i].item(),
                        'margin': margins[i].item(),
                        'predicted_label': predicted_labels[i].item()
                    })

        # Валидация
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, _, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion_none(logits, labels).mean()
                epoch_val_loss += loss.item() * inputs.size(0)

        avg_train_loss = epoch_train_loss / len(train_dataset)
        avg_val_loss = epoch_val_loss / len(test_dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Эпоха {epoch}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Сохранение результатов
    os.makedirs('outputs', exist_ok=True)
    df_traj = pd.DataFrame(trajectories)
    df_traj.to_csv('outputs/raw_trajectories.csv', index=False)
    print("Траектории сохранены в 'outputs/raw_trajectories.csv'")

    os.makedirs('plots', exist_ok=True)
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('plots/loss_curve.png')
    plt.close()
    print("График лосса сохранен в 'plots/loss_curve.png'")


if __name__ == '__main__':
    main()