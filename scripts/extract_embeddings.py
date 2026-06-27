import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# Добавляем корень проекта в sys.path
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


def main():
    parser = argparse.ArgumentParser(description="Extract ImageNet embeddings using ResNet-18")
    parser.add_argument('--smoke_test', action='store_true', help='Process only the first 200 samples')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for extraction')
    args = parser.parse_args()

    # Определение устройства (с приоритетом для Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Используемое устройство: {device}")

    # Трансформации: для ImageNet-предобученных моделей нужна стандартная нормализация
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Загрузка датасета CIFAR-10N...")
    base_dataset = CIFAR10N(root='./data', noise_type='aggregate', train=True, transform=transform, download=False)
    dataset = IndexedDataset(base_dataset)

    if args.smoke_test:
        print("ВНИМАНИЕ: Запуск в режиме smoke_test (200 примеров).")
        dataset = Subset(dataset, range(200))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("Инициализация предобученного ResNet-18 (ImageNet)...")
    # Загружаем модель с весами ImageNet
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Замораживаем все веса
    for param in model.parameters():
        param.requires_grad = False

    # Заменяем финальный слой на Identity, чтобы получать вектор размера 512
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    all_sample_ids = []
    all_embeddings = []

    print("Извлечение эмбеддингов...")
    with torch.no_grad():
        for sample_ids, inputs, _, _ in tqdm(dataloader, desc="Processing batches"):
            inputs = inputs.to(device)

            # Получаем фичи
            embeddings = model(inputs)

            # Переносим на CPU для сохранения
            all_sample_ids.append(sample_ids.cpu())
            all_embeddings.append(embeddings.cpu())

    # Объединяем батчи в единые тензоры
    final_sample_ids = torch.cat(all_sample_ids, dim=0)
    final_embeddings = torch.cat(all_embeddings, dim=0)

    # Сохраняем результат
    os.makedirs('outputs', exist_ok=True)
    output_path = 'outputs/embeddings.pt'

    torch.save({
        'sample_ids': final_sample_ids,
        'embeddings': final_embeddings
    }, output_path)

    print(f"\nГотово! Эмбеддинги успешно сохранены в {output_path}")
    print(f"Размерность тензора эмбеддингов: {final_embeddings.shape}")


if __name__ == '__main__':
    main()