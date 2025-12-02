import os
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config.config import DataConfig, ModelConfig, TrainingConfig, DeviceConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA AUGMENTATION PIPELINES
# ============================================================================

def get_train_transforms():
    """
    Training augmentation pipeline
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop((ModelConfig.INPUT_HEIGHT, ModelConfig.INPUT_WIDTH)),
        transforms.RandomHorizontalFlip(p=DataConfig.RANDOM_HORIZONTAL_FLIP),
        transforms.RandomRotation(degrees=DataConfig.RANDOM_ROTATION),
        transforms.ColorJitter(
            brightness=DataConfig.RANDOM_BRIGHTNESS,
            contrast=DataConfig.RANDOM_CONTRAST
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=DataConfig.NORMALIZE_MEAN,
            std=DataConfig.NORMALIZE_STD
        )
    ])


def get_val_test_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((ModelConfig.INPUT_HEIGHT, ModelConfig.INPUT_WIDTH)),  # Center crop to 224x224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=DataConfig.NORMALIZE_MEAN,
            std=DataConfig.NORMALIZE_STD
        )
    ])


# ============================================================================
# CLASS LABEL MANAGEMENT
# ============================================================================

class LabelEncoder:
    def __init__(self, class_names):
        self.class_names = sorted(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_names)

    def encode(self, class_name):
        # class name -> index
        if class_name not in self.class_to_idx:
            raise ValueError(f"Unknown class name: {class_name}")
        return self.class_to_idx[class_name]

    def decode(self, class_idx):
        # class index -> name
        if class_idx not in self.idx_to_class:
            raise ValueError(f"Unknown class index: {class_idx}")
        return self.idx_to_class[class_idx]

    def save(self, filepath):
        # save label encoder to JSON file.
        data = {
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': {str(k): v for k, v in self.idx_to_class.items()},
            'num_classes': self.num_classes
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Label encoder saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        encoder = cls(data['class_names'])
        logger.info(f"Label encoder loaded from {filepath}")
        return encoder


# ============================================================================
# SPLIT MANAGEMENT
# ============================================================================

def create_splits(data_root, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    # validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    # set random seed
    random.seed(seed)
    np.random.seed(seed)

    # get all class directories
    class_dirs = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])

    splits = {'train': [], 'val': [], 'test': []}

    # for each class, split its images
    for class_name in class_dirs:
        class_path = os.path.join(data_root, class_name)

        # get all image files
        image_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(tuple(f".{fmt}" for fmt in DataConfig.SUPPORTED_FORMATS))
        ]

        # shuffle images
        random.shuffle(image_files)

        # calculate split indices
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # split files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]

        # splits with full paths
        splits['train'].extend([(class_name, f) for f in train_files])
        splits['val'].extend([(class_name, f) for f in val_files])
        splits['test'].extend([(class_name, f) for f in test_files])

    # Save splits
    splits_dir = DataConfig.SPLITS_PATH
    os.makedirs(splits_dir, exist_ok=True)

    for split_name, files in splits.items():
        split_file = os.path.join(splits_dir, f'{split_name}_split.json')
        with open(split_file, 'w') as f:
            json.dump(files, f, indent=2)
        logger.info(f"{split_name} split: {len(files)} images saved to {split_file}")

    return splits


def load_splits(splits_path):
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_file = os.path.join(splits_path, f'{split_name}_split.json')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            splits[split_name] = json.load(f)
        logger.info(f"{split_name} split: {len(splits[split_name])} images loaded")

    return splits


# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================

class AwA2Dataset(Dataset):
    """
      Loads PNG images from disk with proper error handling
      Applies transforms (resize to 224*224, normalize to ImageNet stats)
      Returns (image tensor, label) tuples
      Handles class name → label index mapping
      Supports train/val/test split filtering
    """

    def __init__(self, root_dir, split="train", transform=None, use_split_files=False):
        self.root_dir = root_dir
        self.split = split.lower()
        self.use_split_files = use_split_files

        # Validate split argument
        if self.split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")

        # Initialize samples and classes
        if use_split_files:
            self._load_from_split_files()
        else:
            self._load_from_directories()

        # Create label encoder
        self.label_encoder = LabelEncoder(self.classes)

        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            if self.split == "train":
                self.transform = get_train_transforms()
            else:
                self.transform = get_val_test_transforms()

    def _load_from_directories(self):
        # split directory (e.g., ./data/raw/train)
        self.split_dir = os.path.join(self.root_dir, self.split)

        # verify split directory exists
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # List all class folders and sort them for consistent label assignment
        self.classes = sorted([
            d for d in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, d))
        ])

        if len(self.classes) == 0:
            raise ValueError(f"No class directories found in {self.split_dir}")

        # Gather all image file paths + labels
        self.samples = []
        supported_formats = tuple(f".{fmt}" for fmt in DataConfig.SUPPORTED_FORMATS)

        for cls in self.classes:
            cls_folder = os.path.join(self.split_dir, cls)
            cls_idx = self.classes.index(cls)

            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(supported_formats):
                    img_path = os.path.join(cls_folder, fname)
                    self.samples.append((img_path, cls_idx))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.split_dir}")

    def _load_from_split_files(self):
        splits = load_splits(DataConfig.SPLITS_PATH)
        split_data = splits[self.split]

        # extract unique class names
        self.classes = sorted(list(set([class_name for class_name, _ in split_data])))

        # build samples list
        self.samples = []
        for class_name, filename in split_data:
            img_path = os.path.join(self.root_dir, class_name, filename)
            cls_idx = self.classes.index(class_name)
            self.samples.append((img_path, cls_idx))

        if len(self.samples) == 0:
            raise ValueError(f"No images found for {self.split} split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")

        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error loading image {img_path}: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading image {img_path}: {str(e)}")

        # apply transforms
        if self.transform is not None:
            try:
                image = self.transform(image)
            except Exception as e:
                raise RuntimeError(f"Error applying transform to image {img_path}: {str(e)}")

        return image, label

    def get_class_name(self, idx):
        return self.label_encoder.decode(idx)

    def get_num_classes(self):
        return self.label_encoder.num_classes


# ============================================================================
# DATA VALIDATION & DEBUGGING
# ============================================================================

def validate_dataset(dataset, num_samples=10):
    logger.info(f"Validating dataset: {dataset.split} split")

    stats = {
        'total_samples': len(dataset),
        'num_classes': dataset.get_num_classes(),
        'corrupted_images': [],
        'min_pixel_values': [],
        'max_pixel_values': [],
        'mean_pixel_values': [],
        'image_dimensions': []
    }

    # check random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        try:
            image, label = dataset[idx]

            # Check if image is a tensor
            if not torch.is_tensor(image):
                logger.warning(f"Image at index {idx} is not a tensor")
                continue

            # Compute statistics
            stats['min_pixel_values'].append(image.min().item())
            stats['max_pixel_values'].append(image.max().item())
            stats['mean_pixel_values'].append(image.mean().item())
            stats['image_dimensions'].append(tuple(image.shape))

        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            stats['corrupted_images'].append(idx)

    # Compute averages
    if stats['min_pixel_values']:
        stats['avg_min'] = np.mean(stats['min_pixel_values'])
        stats['avg_max'] = np.mean(stats['max_pixel_values'])
        stats['avg_mean'] = np.mean(stats['mean_pixel_values'])

    # Log statistics
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Number of classes: {stats['num_classes']}")
    logger.info(f"Corrupted images: {len(stats['corrupted_images'])}")
    if stats['min_pixel_values']:
        logger.info(f"Pixel value range: [{stats['avg_min']:.3f}, {stats['avg_max']:.3f}]")
        logger.info(f"Average pixel value: {stats['avg_mean']:.3f}")

    return stats


def visualize_batch(dataloader, num_images=16, save_path=None):
    try:
        import matplotlib.pyplot as plt

        # Get a batch
        images, labels = next(iter(dataloader))

        # Denormalize images
        mean = torch.tensor(DataConfig.NORMALIZE_MEAN).view(3, 1, 1)
        std = torch.tensor(DataConfig.NORMALIZE_STD).view(3, 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0, 1)

        # Create grid
        n_rows = int(np.sqrt(num_images))
        n_cols = int(np.ceil(num_images / n_rows))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(min(num_images, len(images))):
            img = images[i].permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            axes[i].set_title(f"Label: {labels[i].item()}")
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()

    except ImportError:
        logger.warning("matplotlib not installed. Skipping visualization.")


# ============================================================================
# DATALOADER FACTORY FUNCTION
# ============================================================================

def seed_worker(worker_id):
    """
    Seed worker processes for reproducible data loading across different GPUs.

    This ensures that DataLoader workers use consistent random seeds,
    preventing batch composition variance between different hardware MX450 vs T4.

    Critical for reproducibility when using:
      Random augmentations (rotation, crop, flip, color jitter)
      Multiple worker processes (num_workers > 0)
      Different GPUs with different worker scheduling
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(data_root, batch_size=None, num_workers=None, use_split_files=False, phase=1):
    # get batch size from config if not provided
    if batch_size is None:
        batch_size = TrainingConfig.PHASE1_BATCH_SIZE if phase == 1 else TrainingConfig.PHASE2_BATCH_SIZE

    if num_workers is None:
        num_workers = DataConfig.NUM_WORKERS

    # Create datasets
    train_dataset = AwA2Dataset(
        root_dir=data_root,
        split='train',
        transform=get_train_transforms(),
        use_split_files=use_split_files
    )

    val_dataset = AwA2Dataset(
        root_dir=data_root,
        split='val',
        transform=get_val_test_transforms(),
        use_split_files=use_split_files
    )

    test_dataset = AwA2Dataset(
        root_dir=data_root,
        split='test',
        transform=get_val_test_transforms(),
        use_split_files=use_split_files
    )

    # create generator for reproducibility across GPUs
    g = torch.Generator()
    g.manual_seed(DeviceConfig.SEED)

    # Get optimized dataloader settings
    prefetch_factor = getattr(DataConfig, 'PREFETCH_FACTOR', 2)
    persistent_workers = getattr(DataConfig, 'PERSISTENT_WORKERS', False) and num_workers > 0

    # Create DataLoaders with worker seeding for GPU-independent reproducibility
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=DataConfig.SHUFFLE_TRAIN,
        num_workers=num_workers,
        pin_memory=DataConfig.PIN_MEMORY,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=DataConfig.SHUFFLE_VAL,
        num_workers=num_workers,
        pin_memory=DataConfig.PIN_MEMORY,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=DataConfig.SHUFFLE_TEST,
        num_workers=num_workers,
        pin_memory=DataConfig.PIN_MEMORY,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g
    )

    logger.info(f"DataLoaders created:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_class_distribution(dataset):
    class_counts = {}
    for _, label in dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1

    logger.info(f"Class distribution for {dataset.split} split:")
    for class_idx, count in sorted(class_counts.items()):
        class_name = dataset.get_class_name(class_idx)
        logger.info(f"  {class_name} (idx {class_idx}): {count} samples")

    return class_counts


def compute_class_weights(dataset):
    class_counts = get_class_distribution(dataset)
    total_samples = len(dataset)
    num_classes = dataset.get_num_classes()

    # Compute weights inverse frequency
    weights = torch.zeros(num_classes)
    for class_idx, count in class_counts.items():
        weights[class_idx] = total_samples / (num_classes * count)

    logger.info(f"Class weights computed: min={weights.min():.3f}, max={weights.max():.3f}")

    return weights
