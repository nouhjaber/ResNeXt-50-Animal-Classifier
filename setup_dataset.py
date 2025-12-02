"""
Usage:
    python setup_dataset.py --source Animals_with_Attributes2 --dest data/raw
    python setup_dataset.py --source Animals_with_Attributes2 --dest data/raw --mode zero_shot
    python setup_dataset.py --source Animals_with_Attributes2 --dest data/raw --train_ratio 0.7 --val_ratio 0.15
"""

import os
import shutil
import random
import argparse
import json
from pathlib import Path
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Setup Animals with Attributes 2 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--source',
        type=str,
        default='Animals_with_Attributes2',
        help='Path to source AwA2 dataset directory (default: Animals_with_Attributes2)'
    )

    parser.add_argument(
        '--dest',
        type=str,
        default='data/raw',
        help='Path to destination directory (default: data/raw)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['standard', 'zero_shot'],
        default='standard',
        help='Dataset split mode: standard (all 50 classes split 70/15/15) or zero_shot (40 train classes, 10 test classes)'
    )

    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )

    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )

    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--copy',
        action='store_true',
        default=True,
        help='Copy files instead of moving them (default: True)'
    )

    parser.add_argument(
        '--no_copy',
        action='store_false',
        dest='copy',
        help='Move files instead of copying them'
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be done without actually copying/moving files'
    )

    parser.add_argument(
        '--save_splits',
        action='store_true',
        help='Save split information to JSON files in data/splits/'
    )

    return parser.parse_args()


def load_class_list(filepath):
    """Load class names from text file"""
    with open(filepath, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes


def get_image_files(directory, extensions=None):
    """Get all image files in a directory"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    image_files = []
    for ext in extensions:
        image_files.extend([f for f in os.listdir(directory) if f.endswith(ext)])

    return sorted(image_files)


def split_images(image_files, train_ratio, val_ratio, test_ratio, seed=42):
    """Split image files into train, val, test sets"""
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total <= 1.01):  # Allow small floating point errors
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    # Shuffle images with fixed seed
    random.seed(seed)
    shuffled_images = image_files.copy()
    random.shuffle(shuffled_images)

    # Calculate split indices
    n_total = len(shuffled_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_images = shuffled_images[:n_train]
    val_images = shuffled_images[n_train:n_train + n_val]
    test_images = shuffled_images[n_train + n_val:]

    return train_images, val_images, test_images


def setup_standard_mode(args):
    """
    Standard mode: Use all 50 classes, split each class 70/15/15
    """
    logger.info("=" * 80)
    logger.info("STANDARD MODE: Splitting all 50 classes into train/val/test")
    logger.info("=" * 80)

    # paths
    source_images_dir = os.path.join(args.source, 'JPEGImages')

    if not os.path.exists(source_images_dir):
        raise FileNotFoundError(f"Source images directory not found: {source_images_dir}")

    # get all class directories
    all_classes = sorted([
        d for d in os.listdir(source_images_dir)
        if os.path.isdir(os.path.join(source_images_dir, d))
    ])

    logger.info(f"Found {len(all_classes)} classes")
    logger.info(f"Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")

    # Statistics
    stats = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int)
    }

    split_info = {
        'train': [],
        'val': [],
        'test': []
    }

    # Process each class
    for class_name in all_classes:
        source_class_dir = os.path.join(source_images_dir, class_name)

        # Get all images for this class
        image_files = get_image_files(source_class_dir)

        if len(image_files) == 0:
            logger.warning(f"No images found for class: {class_name}")
            continue

        logger.info(f"Processing {class_name}: {len(image_files)} images")

        # Split images
        train_images, val_images, test_images = split_images(
            image_files,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed
        )

        # Copy/move images to destination
        for split_name, images_list in [('train', train_images), ('val', val_images), ('test', test_images)]:
            dest_class_dir = os.path.join(args.dest, split_name, class_name)

            if not args.dry_run:
                os.makedirs(dest_class_dir, exist_ok=True)

            for image_file in images_list:
                source_path = os.path.join(source_class_dir, image_file)
                dest_path = os.path.join(dest_class_dir, image_file)

                if not args.dry_run:
                    if args.copy:
                        shutil.copy2(source_path, dest_path)
                    else:
                        shutil.move(source_path, dest_path)

                stats[split_name][class_name] += 1
                split_info[split_name].append((class_name, image_file))

        logger.info(f"  -> Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

    # Print summary
    print_summary(stats, all_classes)

    # Save split information
    if args.save_splits and not args.dry_run:
        save_split_info(split_info, 'data/splits')

    return stats


def setup_zero_shot_mode(args):
    """
    Zero-shot mode: 40 train classes (split into train/val), 10 test classes
    """
    logger.info("=" * 80)
    logger.info("ZERO-SHOT MODE: Using predefined train/test class splits")
    logger.info("=" * 80)

    # Paths
    source_images_dir = os.path.join(args.source, 'JPEGImages')
    trainclasses_file = os.path.join(args.source, 'trainclasses.txt')
    testclasses_file = os.path.join(args.source, 'testclasses.txt')

    # Validate paths
    if not os.path.exists(source_images_dir):
        raise FileNotFoundError(f"Source images directory not found: {source_images_dir}")
    if not os.path.exists(trainclasses_file):
        raise FileNotFoundError(f"Train classes file not found: {trainclasses_file}")
    if not os.path.exists(testclasses_file):
        raise FileNotFoundError(f"Test classes file not found: {testclasses_file}")

    # Load class lists
    train_classes = load_class_list(trainclasses_file)
    test_classes = load_class_list(testclasses_file)

    logger.info(f"Train classes: {len(train_classes)}")
    logger.info(f"Test classes: {len(test_classes)}")

    # Statistics
    stats = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int)
    }

    split_info = {
        'train': [],
        'val': [],
        'test': []
    }

    # Process train classes (split into train/val)
    logger.info("Processing train classes (splitting into train/val)...")
    for class_name in train_classes:
        source_class_dir = os.path.join(source_images_dir, class_name)

        if not os.path.exists(source_class_dir):
            logger.warning(f"Class directory not found: {class_name}")
            continue

        # Get all images
        image_files = get_image_files(source_class_dir)

        if len(image_files) == 0:
            logger.warning(f"No images found for class: {class_name}")
            continue

        logger.info(f"Processing {class_name}: {len(image_files)} images")

        # Split into train/val (no test for train classes in zero-shot mode)
        # Adjust ratios: train_ratio / (train_ratio + val_ratio)
        adjusted_train_ratio = args.train_ratio / (args.train_ratio + args.val_ratio)
        adjusted_val_ratio = args.val_ratio / (args.train_ratio + args.val_ratio)

        train_images, val_images, _ = split_images(
            image_files,
            adjusted_train_ratio,
            adjusted_val_ratio,
            0.0,  # No test split for train classes
            args.seed
        )

        # Copy images
        for split_name, images_list in [('train', train_images), ('val', val_images)]:
            dest_class_dir = os.path.join(args.dest, split_name, class_name)

            if not args.dry_run:
                os.makedirs(dest_class_dir, exist_ok=True)

            for image_file in images_list:
                source_path = os.path.join(source_class_dir, image_file)
                dest_path = os.path.join(dest_class_dir, image_file)

                if not args.dry_run:
                    if args.copy:
                        shutil.copy2(source_path, dest_path)
                    else:
                        shutil.move(source_path, dest_path)

                stats[split_name][class_name] += 1
                split_info[split_name].append((class_name, image_file))

        logger.info(f"  -> Train: {len(train_images)}, Val: {len(val_images)}")

    # Process test classes
    logger.info("Processing test classes (all images to test set)...")
    for class_name in test_classes:
        source_class_dir = os.path.join(source_images_dir, class_name)

        if not os.path.exists(source_class_dir):
            logger.warning(f"Class directory not found: {class_name}")
            continue

        image_files = get_image_files(source_class_dir)

        if len(image_files) == 0:
            logger.warning(f"No images found for class: {class_name}")
            continue

        logger.info(f"Processing {class_name}: {len(image_files)} images")

        dest_class_dir = os.path.join(args.dest, 'test', class_name)

        if not args.dry_run:
            os.makedirs(dest_class_dir, exist_ok=True)

        for image_file in image_files:
            source_path = os.path.join(source_class_dir, image_file)
            dest_path = os.path.join(dest_class_dir, image_file)

            if not args.dry_run:
                if args.copy:
                    shutil.copy2(source_path, dest_path)
                else:
                    shutil.move(source_path, dest_path)

            stats['test'][class_name] += 1
            split_info['test'].append((class_name, image_file))

        logger.info(f"  -> Test: {len(image_files)}")

    # Print summary
    all_classes = train_classes + test_classes
    print_summary(stats, all_classes)

    # Save split information
    if args.save_splits and not args.dry_run:
        save_split_info(split_info, 'data/splits')

    return stats


def print_summary(stats, all_classes):
    """Print dataset statistics"""
    logger.info("=" * 80)
    logger.info("DATASET SETUP COMPLETE")
    logger.info("=" * 80)

    total_train = sum(stats['train'].values())
    total_val = sum(stats['val'].values())
    total_test = sum(stats['test'].values())
    total_all = total_train + total_val + total_test

    logger.info(f"Total classes: {len(all_classes)}")
    logger.info(f"Total images: {total_all}")
    logger.info("")
    logger.info(f"Train set: {total_train} images ({total_train/total_all*100:.1f}%)")
    logger.info(f"  Classes: {len(stats['train'])}")
    logger.info(f"Val set: {total_val} images ({total_val/total_all*100:.1f}%)")
    logger.info(f"  Classes: {len(stats['val'])}")
    logger.info(f"Test set: {total_test} images ({total_test/total_all*100:.1f}%)")
    logger.info(f"  Classes: {len(stats['test'])}")
    logger.info("")

    # Per-class statistics
    logger.info("Per-class distribution:")
    logger.info(f"{'Class':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    logger.info("-" * 80)

    for class_name in sorted(all_classes):
        train_count = stats['train'].get(class_name, 0)
        val_count = stats['val'].get(class_name, 0)
        test_count = stats['test'].get(class_name, 0)
        total_count = train_count + val_count + test_count

        logger.info(f"{class_name:<25} {train_count:>8} {val_count:>8} {test_count:>8} {total_count:>8}")

    logger.info("=" * 80)


def save_split_info(split_info, output_dir):
    """Save split information to JSON files"""
    os.makedirs(output_dir, exist_ok=True)

    for split_name, items in split_info.items():
        output_file = os.path.join(output_dir, f'{split_name}_split.json')
        with open(output_file, 'w') as f:
            json.dump(items, f, indent=2)
        logger.info(f"Saved {split_name} split info to: {output_file}")


def verify_setup(dest_dir):
    """Verify the dataset setup"""
    logger.info("=" * 80)
    logger.info("VERIFYING DATASET SETUP")
    logger.info("=" * 80)

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dest_dir, split)

        if not os.path.exists(split_dir):
            logger.error(f"Split directory not found: {split_dir}")
            continue

        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        total_images = 0

        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            images = get_image_files(class_dir)
            total_images += len(images)

        logger.info(f"{split:5}: {len(classes)} classes, {total_images} images")

    logger.info("=" * 80)


def main():
    """Main function"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Animals with Attributes 2 Dataset Setup")
    logger.info("=" * 80)
    logger.info(f"Source: {args.source}")
    logger.info(f"Destination: {args.dest}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Action: {'COPY' if args.copy else 'MOVE'}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 80)

    if args.dry_run:
        logger.warning("DRY RUN MODE - No files will be copied/moved")

    if not os.path.exists(args.source):
        logger.error(f"Source directory not found: {args.source}")
        return 1

    # Create destination directory
    if not args.dry_run:
        os.makedirs(args.dest, exist_ok=True)

    # Setup dataset based on mode
    try:
        if args.mode == 'standard':
            stats = setup_standard_mode(args)
        elif args.mode == 'zero_shot':
            stats = setup_zero_shot_mode(args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return 1

        # Verify setup
        if not args.dry_run:
            verify_setup(args.dest)

        logger.info("Dataset setup complete!")
        logger.info(f"You can now train the model with: python train.py --data_root {args.dest}")

        return 0

    except Exception as e:
        logger.error(f"Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
