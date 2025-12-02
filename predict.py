"""
ResNeXt-50 Inference Script for AwA2 Dataset

This script performs inference on single images, batches of images, or test sets.

Usage Examples:
    # Single image prediction
    python predict.py --image ./test_image.png --checkpoint ./models/checkpoints/best_model.pth

    # Batch prediction on directory
    python predict.py --image_dir ./test_images/ --checkpoint ./models/checkpoints/best_model.pth --output results.csv

    # Prediction with visualization
    python predict.py --image ./test_image.png --checkpoint ./models/checkpoints/best_model.pth --visualize --output_dir ./predictions/

    # Top-K predictions
    python predict.py --image ./test_image.png --checkpoint ./models/checkpoints/best_model.pth --top_k 5
"""

import os
import sys
import argparse
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.config import ModelConfig, DataConfig
from src.models.resnext50 import ResNeXt50
from src.data.dataset import get_val_test_transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Run inference with ResNeXt-50 on AwA2 images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image',
        type=str,
        help='Path to single image file'
    )
    input_group.add_argument(
        '--image_dir',
        type=str,
        help='Path to directory containing images'
    )

    # Model arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=50,
        help='Number of classes (default: 50)'
    )

    # Class names
    parser.add_argument(
        '--class_names',
        type=str,
        default=None,
        help='Path to text file with class names (one per line)'
    )

    # Prediction options
    parser.add_argument(
        '--top_k',
        type=int,
        default=1,
        help='Return top K predictions (default: 1)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Minimum confidence threshold (default: 0.0)'
    )

    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file for batch predictions'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./predictions',
        help='Output directory for visualizations (default: ./predictions)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization of predictions'
    )
    parser.add_argument(
        '--save_probs',
        action='store_true',
        help='Save full probability distribution'
    )

    # Device arguments
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto)'
    )

    # Other arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for directory inference (default: 32)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    return parser.parse_args()


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(checkpoint_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    Load model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        num_classes: Number of output classes
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create model
    model = ResNeXt50(num_classes=num_classes)

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
            if 'best_val_acc' in checkpoint:
                logger.info(f"Model validation accuracy: {checkpoint['best_val_acc']:.2f}%")
        else:
            model.load_state_dict(checkpoint)

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully!")

    return model


def load_class_names(class_names_path: Optional[str], num_classes: int) -> List[str]:
    """
    Load class names from file or generate default names

    Args:
        class_names_path: Path to class names file (optional)
        num_classes: Number of classes

    Returns:
        List of class names
    """
    if class_names_path and os.path.exists(class_names_path):
        logger.info(f"Loading class names from: {class_names_path}")
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]

        if len(class_names) != num_classes:
            logger.warning(
                f"Number of class names ({len(class_names)}) doesn't match "
                f"num_classes ({num_classes}). Using indices instead."
            )
            return [f"Class_{i}" for i in range(num_classes)]

        return class_names
    else:
        logger.info("Using default class names (Class_0, Class_1, ...)")
        return [f"Class_{i}" for i in range(num_classes)]


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocess single image for inference

    Args:
        image_path: Path to image file

    Returns:
        Preprocessed image tensor
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Check if valid image format
    valid_formats = tuple(f".{fmt}" for fmt in DataConfig.SUPPORTED_FORMATS)
    if not image_path.lower().endswith(valid_formats):
        raise ValueError(
            f"Unsupported image format. Supported: {DataConfig.SUPPORTED_FORMATS}"
        )

    try:
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transforms (same as validation/test)
        transform = get_val_test_transforms()
        image_tensor = transform(image)

        return image_tensor

    except Exception as e:
        raise RuntimeError(f"Failed to preprocess image {image_path}: {e}")


def load_images_from_directory(directory: str) -> List[Tuple[str, str]]:
    """
    Load all valid images from directory

    Args:
        directory: Path to directory

    Returns:
        List of (image_path, filename) tuples
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")

    valid_formats = tuple(f".{fmt}" for fmt in DataConfig.SUPPORTED_FORMATS)
    image_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_formats):
                image_path = os.path.join(root, file)
                image_files.append((image_path, file))

    if len(image_files) == 0:
        raise ValueError(f"No valid images found in {directory}")

    logger.info(f"Found {len(image_files)} images in {directory}")

    return image_files


# ============================================================================
# INFERENCE
# ============================================================================

def predict_single_image(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    top_k: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on single image

    Args:
        model: PyTorch model
        image_tensor: Preprocessed image tensor
        device: Device
        top_k: Number of top predictions to return

    Returns:
        Tuple of (top_k_indices, top_k_probabilities)
    """
    # Add batch dimension and move to device
    image_batch = image_tensor.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(image_batch)
        probabilities = torch.softmax(outputs, dim=1)

        # Get top-k predictions
        top_k_probs, top_k_indices = probabilities.topk(top_k, dim=1)

        # Convert to numpy
        top_k_probs = top_k_probs.cpu().numpy()[0]
        top_k_indices = top_k_indices.cpu().numpy()[0]

    return top_k_indices, top_k_probs


def predict_batch(
    model: nn.Module,
    image_paths: List[str],
    device: torch.device,
    batch_size: int = 32,
    top_k: int = 1
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Run inference on batch of images

    Args:
        model: PyTorch model
        image_paths: List of image paths
        device: Device
        batch_size: Batch size for processing
        top_k: Number of top predictions to return

    Returns:
        List of (top_k_indices, top_k_probabilities) for each image
    """
    results = []

    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]

        # Preprocess batch
        batch_tensors = []
        for img_path in batch_paths:
            try:
                tensor = preprocess_image(img_path)
                batch_tensors.append(tensor)
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                # Add dummy result
                results.append((np.array([-1]), np.array([0.0])))
                continue

        if len(batch_tensors) == 0:
            continue

        # Stack into batch
        batch = torch.stack(batch_tensors).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(batch)
            probabilities = torch.softmax(outputs, dim=1)

            # Get top-k predictions
            top_k_probs, top_k_indices = probabilities.topk(top_k, dim=1)

            # Convert to numpy
            top_k_probs = top_k_probs.cpu().numpy()
            top_k_indices = top_k_indices.cpu().numpy()

            # Add to results
            for j in range(len(batch_tensors)):
                results.append((top_k_indices[j], top_k_probs[j]))

    return results


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_prediction(
    indices: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str],
    threshold: float = 0.0
) -> List[Dict[str, any]]:
    """
    Format prediction results

    Args:
        indices: Predicted class indices
        probabilities: Prediction probabilities
        class_names: List of class names
        threshold: Minimum confidence threshold

    Returns:
        List of prediction dictionaries
    """
    predictions = []

    for idx, prob in zip(indices, probabilities):
        if prob >= threshold:
            predictions.append({
                'class_index': int(idx),
                'class_name': class_names[idx] if idx < len(class_names) else f"Class_{idx}",
                'confidence': float(prob)
            })

    return predictions


def print_prediction(predictions: List[Dict], image_path: str):
    """Print prediction results to console"""
    logger.info("=" * 60)
    logger.info(f"Prediction for: {image_path}")
    logger.info("=" * 60)

    for i, pred in enumerate(predictions, 1):
        logger.info(
            f"  {i}. {pred['class_name']} "
            f"(confidence: {pred['confidence']:.4f})"
        )

    logger.info("=" * 60)


def save_predictions_csv(
    results: List[Tuple[str, List[Dict]]],
    output_path: str
):
    """
    Save batch predictions to CSV file

    Args:
        results: List of (filename, predictions) tuples
        output_path: Path to output CSV file
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed. Cannot save CSV.")
        return

    # Prepare data
    data = []
    for filename, predictions in results:
        if len(predictions) > 0:
            pred = predictions[0]  # Top prediction
            data.append({
                'filename': filename,
                'predicted_class': pred['class_name'],
                'class_index': pred['class_index'],
                'confidence': pred['confidence']
            })
        else:
            data.append({
                'filename': filename,
                'predicted_class': 'ERROR',
                'class_index': -1,
                'confidence': 0.0
            })

    # Create DataFrame and save
    df = pd.DataFrame(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Predictions saved to: {output_path}")


def save_predictions_json(
    results: List[Tuple[str, List[Dict]]],
    output_path: str
):
    """
    Save batch predictions to JSON file

    Args:
        results: List of (filename, predictions) tuples
        output_path: Path to output JSON file
    """
    import json

    # Prepare data
    data = {}
    for filename, predictions in results:
        data[filename] = predictions

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Predictions saved to: {output_path}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_prediction(
    image_path: str,
    predictions: List[Dict],
    output_path: str
):
    """
    Visualize prediction on image

    Args:
        image_path: Path to original image
        predictions: List of prediction dictionaries
        output_path: Path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        logger.warning("matplotlib not installed. Skipping visualization.")
        return

    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image_np)
    ax.axis('off')

    # Add prediction text
    text_str = ""
    for i, pred in enumerate(predictions[:3], 1):  # Show top 3
        text_str += f"{i}. {pred['class_name']}\n   ({pred['confidence']:.2%})\n"

    # Add text box
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(
        0.02, 0.98,
        text_str.strip(),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=props
    )

    # Add title
    title = f"Top Prediction: {predictions[0]['class_name']}"
    plt.title(title, fontsize=14, pad=10)

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualization saved to: {output_path}")


def visualize_batch_predictions(
    results: List[Tuple[str, str, List[Dict]]],
    output_dir: str,
    max_images: int = 20
):
    """
    Create grid visualization for batch predictions

    Args:
        results: List of (image_path, filename, predictions) tuples
        output_dir: Output directory
        max_images: Maximum number of images to visualize
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Skipping visualization.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Limit number of images
    results = results[:max_images]

    # Calculate grid size
    n_images = len(results)
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_images > 1 else [axes]

    # Denormalize function
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx, (img_path, filename, predictions) in enumerate(results):
        try:
            # Load and display image
            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image)

            axes[idx].imshow(image_np)
            axes[idx].axis('off')

            # Add prediction text
            if len(predictions) > 0:
                pred = predictions[0]
                title = f"{pred['class_name']}\n{pred['confidence']:.2%}"
                color = 'green' if pred['confidence'] > 0.7 else 'orange' if pred['confidence'] > 0.5 else 'red'
                axes[idx].set_title(title, fontsize=10, color=color)
            else:
                axes[idx].set_title("ERROR", fontsize=10, color='red')

        except Exception as e:
            logger.warning(f"Failed to visualize {filename}: {e}")
            axes[idx].axis('off')

    # Hide empty subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'batch_predictions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Batch visualization saved to: {output_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main inference function"""
    # Parse arguments
    args = parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Load model
    try:
        model = load_model(args.checkpoint, args.num_classes, device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Load class names
    class_names = load_class_names(args.class_names, args.num_classes)

    # ========== SINGLE IMAGE PREDICTION ==========
    if args.image:
        logger.info(f"Running inference on single image: {args.image}")

        try:
            # Preprocess
            image_tensor = preprocess_image(args.image)

            # Predict
            indices, probabilities = predict_single_image(
                model, image_tensor, device, top_k=args.top_k
            )

            # Format results
            predictions = format_prediction(
                indices, probabilities, class_names, args.threshold
            )

            # Print results
            print_prediction(predictions, args.image)

            # Visualize if requested
            if args.visualize:
                output_path = os.path.join(
                    args.output_dir,
                    f"{Path(args.image).stem}_prediction.png"
                )
                visualize_prediction(args.image, predictions, output_path)

            # Save to JSON if output specified
            if args.output:
                save_predictions_json([(args.image, predictions)], args.output)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            sys.exit(1)

    # ========== BATCH PREDICTION ==========
    elif args.image_dir:
        logger.info(f"Running batch inference on directory: {args.image_dir}")

        try:
            # Load images
            image_files = load_images_from_directory(args.image_dir)
            image_paths = [path for path, _ in image_files]
            filenames = [name for _, name in image_files]

            # Run batch prediction
            logger.info(f"Processing {len(image_paths)} images...")

            batch_results = predict_batch(
                model, image_paths, device,
                batch_size=args.batch_size,
                top_k=args.top_k
            )

            # Format results
            all_predictions = []
            for i, (indices, probabilities) in enumerate(batch_results):
                predictions = format_prediction(
                    indices, probabilities, class_names, args.threshold
                )
                all_predictions.append((filenames[i], predictions))

            # Save predictions
            if args.output:
                if args.output.endswith('.json'):
                    save_predictions_json(all_predictions, args.output)
                else:
                    save_predictions_csv(all_predictions, args.output)

            # Visualize if requested
            if args.visualize:
                viz_results = [
                    (image_paths[i], filenames[i], pred)
                    for i, (_, pred) in enumerate(all_predictions)
                ]
                visualize_batch_predictions(viz_results, args.output_dir)

            # Print summary
            logger.info("=" * 60)
            logger.info("Batch Prediction Summary")
            logger.info("=" * 60)
            logger.info(f"Total images processed: {len(all_predictions)}")

            # Count predictions per class
            class_counts = {}
            for _, predictions in all_predictions:
                if len(predictions) > 0:
                    class_name = predictions[0]['class_name']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

            logger.info("\nTop predicted classes:")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"  {class_name}: {count}")

            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            sys.exit(1)

    logger.info("Inference complete!")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
