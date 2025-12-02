"""
4 Phase Training
  Phase 1 Layer4 + FC (15 epochs) - Target: 65-70%
      Frozen Stem + Layer1-3
      Trainable Layer4 + FC
      Learning Rates Layer4 (8e-5) FC (3e-4)

  Phase 2 Layer3-4 + FC (25 epochs) - Target: 75-80%
      Frozen Stem + Layer1-2
      Trainable Layer3-4 + FC
      Learning Rates: Layer3 (3e-5) Layer4 (8e-5), FC (3e-4)

  Phase 3 Layer2-4 + FC (25 epochs) - Target: 85-88%
       Frozen Stem + Layer1
       Trainable Layer2-4 + FC
       Learning Rates Layer2 (1e-5) Layer3 (3e-5), Layer4 (8e-5), FC (3e-4)

  Phase 4 ALL layers (30 epochs) - Target: 90-92%
      Frozen None
      Trainable ALL layers
      Learning Rates Stem (1e-6) Layer1 (5e-6), Layer2 (1e-5), Layer3 (3e-5), Layer4 (8e-5), FC (3e-4)

Total 95 epochs across 4 phases

Usage:
    python train.py --data_root ./data/raw --phase 1
    python train.py --data_root ./data/raw --phase 2 --resume ./models/checkpoints/phase1_best_model.pth
    python train.py --data_root ./data/raw --phase all  # Run all 4 phases sequentially
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    DeviceConfig,
    LoggingConfig,
    get_phase_config
)
from src.data.dataset import create_dataloaders
from src.models.resnext50 import create_resnext50
from src.training.trainer import create_trainer
from src.training.evaluator import Evaluator

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
        description='Train ResNeXt-50 on AwA2 Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Data arguments
    parser.add_argument(
        '--data_root',
        type=str,
        default='./data/raw',
        help='Path to dataset root directory (default: ./data/raw)'
    )
    parser.add_argument(
        '--use_split_files',
        action='store_true',
        help='Use pre-generated split files instead of directory structure'
    )

    # Training phase
    parser.add_argument(
        '--phase',
        type=str,
        choices=['1', '2', '3', '4', 'all'],
        default='all',
        help='Training phase: 1 (Layer4+FC), 2 (Layer3-4+FC), 3 (Layer2-4+FC), 4 (ALL layers), or all (default: all)'
    )

    # Phase 1 arguments
    parser.add_argument(
        '--phase1_epochs',
        type=int,
        default=None,
        help=f'Number of epochs for Phase 1 (default: {TrainingConfig.PHASE1_EPOCHS})'
    )
    parser.add_argument(
        '--phase1_batch_size',
        type=int,
        default=None,
        help=f'Batch size for Phase 1 (default: {TrainingConfig.PHASE1_BATCH_SIZE})'
    )

    # Phase 2 arguments
    parser.add_argument(
        '--phase2_epochs',
        type=int,
        default=None,
        help=f'Number of epochs for Phase 2 (default: {TrainingConfig.PHASE2_EPOCHS})'
    )
    parser.add_argument(
        '--phase2_batch_size',
        type=int,
        default=None,
        help=f'Batch size for Phase 2 (default: {TrainingConfig.PHASE2_BATCH_SIZE})'
    )

    # Model arguments
    parser.add_argument(
        '--num_classes',
        type=int,
        default=50,
        help='Number of output classes (default: 50)'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='Use ImageNet pretrained weights (default: True)'
    )
    parser.add_argument(
        '--no_pretrained',
        action='store_false',
        dest='pretrained',
        help='Do not use pretrained weights'
    )

    # Checkpoint arguments
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--fresh-start',
        action='store_true',
        help='Force fresh start (ignore existing checkpoints for auto-resume)'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./models/checkpoints',
        help='Directory to save checkpoints (default: ./models/checkpoints)'
    )

    # Device arguments
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help=f'Number of data loading workers (default: {DataConfig.NUM_WORKERS})'
    )

    # Logging arguments
    parser.add_argument(
        '--tensorboard_dir',
        type=str,
        default='./logs/tensorboard',
        help='TensorBoard log directory (default: ./logs/tensorboard)'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Experiment name for logging (default: auto-generated)'
    )

    # Evaluation arguments
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Only run evaluation on test set (requires --resume)'
    )
    parser.add_argument(
        '--eval_dir',
        type=str,
        default='./results/evaluation',
        help='Directory to save evaluation results (default: ./results/evaluation)'
    )

    # Other arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    return parser.parse_args()


# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def setup_device(args):
    """Setup device for training"""
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    return device


def setup_directories(args):
    """Create necessary directories"""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)

    logger.info("Directories created:")
    logger.info(f"  Checkpoints: {args.checkpoint_dir}")
    logger.info(f"  TensorBoard: {args.tensorboard_dir}")
    logger.info(f"  Evaluation: {args.eval_dir}")


def apply_config_overrides(args):
    """Apply command-line argument overrides to config"""
    if args.phase1_epochs is not None:
        TrainingConfig.PHASE1_EPOCHS = args.phase1_epochs
    if args.phase1_batch_size is not None:
        TrainingConfig.PHASE1_BATCH_SIZE = args.phase1_batch_size

    if args.phase2_epochs is not None:
        TrainingConfig.PHASE2_EPOCHS = args.phase2_epochs
    if args.phase2_batch_size is not None:
        TrainingConfig.PHASE2_BATCH_SIZE = args.phase2_batch_size

    if args.num_workers is not None:
        DataConfig.NUM_WORKERS = args.num_workers

    if args.checkpoint_dir is not None:
        LoggingConfig.CHECKPOINT_DIR = args.checkpoint_dir

    if args.tensorboard_dir is not None:
        LoggingConfig.TENSORBOARD_DIR = args.tensorboard_dir

    if args.experiment_name is not None:
        LoggingConfig.EXPERIMENT_NAME = args.experiment_name


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_phase(model, dataloaders, device, args, phase, resume=False):
    """
    Generic function to train any phase

    Phase 1: Layer4 + FC (15 epochs) - Target: 65-70%
    Phase 2: Layer3-4 + FC (25 epochs) - Target: 75-80%
    Phase 3: Layer2-4 + FC (25 epochs) - Target: 85-88%
    Phase 4: ALL layers (30 epochs) - Target: 90-92%
    """
    from config.config import get_phase_config
    phase_cfg = get_phase_config(phase)

    logger.info("=" * 80)
    logger.info(phase_cfg['description'])
    logger.info(f"Target Accuracy: {phase_cfg['target_accuracy']}")
    logger.info("=" * 80)

    # Create trainer for the phase
    trainer = create_trainer(
        model=model,
        dataloaders=dataloaders,
        phase=phase,
        device=device
    )

    # Check for auto-resume with phase-specific checkpoint
    if not resume and not args.fresh_start:
        latest_checkpoint = os.path.join(args.checkpoint_dir, f'phase{phase}_latest_checkpoint.pth')
        if os.path.exists(latest_checkpoint):
            logger.info("=" * 80)
            logger.info(f"FOUND EXISTING PHASE {phase} CHECKPOINT - AUTO-RESUMING")
            logger.info(f"Checkpoint: {latest_checkpoint}")
            logger.info("=" * 80)
            trainer.load_checkpoint(latest_checkpoint, resume_training=True)
            resume = True

    # print trainable parameters
    if hasattr(model, 'print_architecture_summary'):
        model.print_architecture_summary()

    # Train
    num_epochs = phase_cfg['epochs']
    history = trainer.train(num_epochs=num_epochs, phase=phase, resume=resume)

    logger.info(f"Phase {phase} training complete!")
    logger.info(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    logger.info(f"Target was: {phase_cfg['target_accuracy']}")

    return trainer, history


def run_final_evaluation(model, dataloaders, device, args):
    """Run final evaluation on test set"""
    logger.info("=" * 80)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 80)

    # get class names
    class_names = None
    if hasattr(dataloaders['test'].dataset, 'classes'):
        class_names = dataloaders['test'].dataset.classes

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        device=device,
        class_names=class_names
    )

    # run evaluation
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = os.path.join(args.eval_dir, f'test_evaluation_{timestamp}')

    results = evaluator.evaluate_full(
        dataloader=dataloaders['test'],
        compute_top5=True,
        save_dir=eval_dir
    )

    logger.info("=" * 80)
    logger.info("Final Evaluation Complete!")
    logger.info(f"Results saved to: {eval_dir}")
    logger.info("=" * 80)

    return results


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    # Print banner
    logger.info("=" * 80)
    logger.info("ResNeXt-50 Training on AwA2 Dataset")
    logger.info("=" * 80)

    # Set random seed
    set_random_seed(args.seed)

    # Tesla T4 OPTIMIZATION: Enable cudNN benchmark for faster convolutions
    torch.backends.cudnn.benchmark = True
    logger.info("Tesla T4 OPTIMIZATION: cudNN benchmark mode enabled (auto-tune convolution algorithms)")

    device = setup_device(args)

    setup_directories(args)

    apply_config_overrides(args)

    logger.info("Training Configuration:")
    logger.info(f"  Data root: {args.data_root}")
    logger.info(f"  Phase: {args.phase}")
    logger.info(f"  Number of classes: {args.num_classes}")
    logger.info(f"  Pretrained: {args.pretrained}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Random seed: {args.seed}")

    # ========== LOAD DATA ==========
    logger.info("=" * 80)
    logger.info("Loading Dataset")
    logger.info("=" * 80)

    try:
        if args.phase == '1':
            batch_size = TrainingConfig.PHASE1_BATCH_SIZE
        elif args.phase == '2':
            batch_size = TrainingConfig.PHASE2_BATCH_SIZE
        else:
            # For 'both', use Phase 1 batch size initially
            batch_size = TrainingConfig.PHASE1_BATCH_SIZE

        dataloaders = create_dataloaders(
            data_root=args.data_root,
            batch_size=batch_size,
            num_workers=DataConfig.NUM_WORKERS,
            use_split_files=args.use_split_files,
            phase=1 if args.phase != '2' else 2
        )

        logger.info("Dataset loaded successfully!")
        logger.info(f"  Train samples: {len(dataloaders['train'].dataset)}")
        logger.info(f"  Val samples: {len(dataloaders['val'].dataset)}")
        logger.info(f"  Test samples: {len(dataloaders['test'].dataset)}")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # ========== CREATE MODEL ==========
    logger.info("=" * 80)
    logger.info("Creating Model")
    logger.info("=" * 80)

    try:
        model = create_resnext50(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            freeze_backbone=False  # We'll handle freezing in trainer
        )

        model = model.to(device)
        logger.info("Model created successfully!")

    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise

    # ========== RESUME FROM CHECKPOINT ==========
    if args.resume:
        logger.info("=" * 80)
        logger.info(f"Resuming from checkpoint: {args.resume}")
        logger.info("=" * 80)

        try:
            # GPU-compatible checkpoint loading
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Checkpoint loaded successfully!")

            if 'phase' in checkpoint:
                logger.info(f"  Checkpoint phase: {checkpoint['phase']}")
            if 'epoch' in checkpoint:
                logger.info(f"  Checkpoint epoch: {checkpoint['epoch'] + 1}")
            if 'best_val_acc' in checkpoint:
                logger.info(f"  Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    # ========== EVALUATION ONLY MODE ==========
    if args.eval_only:
        if not args.resume:
            logger.error("--eval_only requires --resume to specify model checkpoint")
            sys.exit(1)

        logger.info("Running evaluation only mode...")
        results = run_final_evaluation(model, dataloaders, device, args)
        logger.info("Evaluation complete!")
        return

    # ========== CHECK FOR FRESH START ==========
    if args.fresh_start:
        logger.info("=" * 80)
        logger.info("FRESH START REQUESTED - Deleting existing checkpoints")
        logger.info("=" * 80)
        import glob
        checkpoint_files = glob.glob(os.path.join(args.checkpoint_dir, '*.pth'))
        for ckpt in checkpoint_files:
            try:
                os.remove(ckpt)
                logger.info(f"Deleted: {ckpt}")
            except Exception as e:
                logger.warning(f"Could not delete {ckpt}: {e}")

    # ========== TRAINING ==========
    try:
        if args.phase in ['1', '2', '3', '4']:
            # Single phase training
            phase = int(args.phase)
            trainer, history = train_phase(model, dataloaders, device, args, phase)

        elif args.phase == 'all':
            # 4-PHASE PROGRESSIVE TRANSFER LEARNING STRATEGY
            logger.info("=" * 80)
            logger.info("STARTING 4-PHASE PROGRESSIVE TRANSFER LEARNING")
            logger.info("  Phase 1: Layer4 + FC (15 epochs) - Target: 65-70%")
            logger.info("  Phase 2: Layer3-4 + FC (25 epochs) - Target: 75-80%")
            logger.info("  Phase 3: Layer2-4 + FC (25 epochs) - Target: 85-88%")
            logger.info("  Phase 4: ALL layers (30 epochs) - Target: 90-92%")
            logger.info("  Total: 95 epochs")
            logger.info("=" * 80)

            # track total training time
            total_training_start = time.time()

            # ========== PHASE 1 Layer4 + FC ==========
            trainer_p1, history_p1 = train_phase(model, dataloaders, device, args, phase=1)

            # Load best Phase 1 model before Phase 2
            logger.info("=" * 80)
            logger.info("PHASE 1 COMPLETE - Loading best checkpoint for Phase 2")
            logger.info("=" * 80)

            best_phase1_checkpoint = os.path.join(args.checkpoint_dir, 'phase1_best_model.pth')
            if os.path.exists(best_phase1_checkpoint):
                checkpoint = torch.load(best_phase1_checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"  Loaded best Phase 1 model")
                logger.info(f"  Best Phase 1 accuracy: {checkpoint['best_val_acc']:.2f}%")
            else:
                logger.warning("Best Phase 1 checkpoint not found! Continuing with current model state.")

            # ========== PHASE 2 Layer3-4 + FC ==========
            trainer_p2, history_p2 = train_phase(model, dataloaders, device, args, phase=2)

            # Load best Phase 2 model before Phase 3
            logger.info("=" * 80)
            logger.info("PHASE 2 COMPLETE - Loading best checkpoint for Phase 3")
            logger.info("=" * 80)

            best_phase2_checkpoint = os.path.join(args.checkpoint_dir, 'phase2_best_model.pth')
            if os.path.exists(best_phase2_checkpoint):
                checkpoint = torch.load(best_phase2_checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"  Loaded best Phase 2 model")
                logger.info(f"  Best Phase 2 accuracy: {checkpoint['best_val_acc']:.2f}%")
            else:
                logger.warning("Best Phase 2 checkpoint not found! Continuing with current model state.")

            # ========== PHASE 3 Layer2-4 + FC ==========
            trainer_p3, history_p3 = train_phase(model, dataloaders, device, args, phase=3)

            # Load best Phase 3 model before Phase 4
            logger.info("=" * 80)
            logger.info("PHASE 3 COMPLETE - Loading best checkpoint for Phase 4")
            logger.info("=" * 80)

            best_phase3_checkpoint = os.path.join(args.checkpoint_dir, 'phase3_best_model.pth')
            if os.path.exists(best_phase3_checkpoint):
                checkpoint = torch.load(best_phase3_checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded best Phase 3 model")
                logger.info(f" Best Phase 3 accuracy: {checkpoint['best_val_acc']:.2f}%")
            else:
                logger.warning("Best Phase 3 checkpoint not found! Continuing with current model state.")

            # ========== PHASE 4: ALL layers ==========
            trainer_p4, history_p4 = train_phase(model, dataloaders, device, args, phase=4)

            # use Phase 4 trainer for final evaluation
            trainer = trainer_p4

            # total training time
            total_training_time = time.time() - total_training_start
            logger.info("=" * 80)
            logger.info("ALL 4 PHASES COMPLETE!")
            logger.info(f"Total training time: {total_training_time/3600:.1f} hours ({total_training_time:.0f}s)")
            logger.info(f"Phase 1 best: {trainer_p1.best_val_acc:.2f}%")
            logger.info(f"Phase 2 best: {trainer_p2.best_val_acc:.2f}%")
            logger.info(f"Phase 3 best: {trainer_p3.best_val_acc:.2f}%")
            logger.info(f"Phase 4 best: {trainer_p4.best_val_acc:.2f}%")
            logger.info("=" * 80)

        else:
            logger.error(f"Invalid phase: {args.phase}")
            sys.exit(1)

        # ========== FINAL EVALUATION ==========
        logger.info("=" * 80)
        logger.info("Training Complete! Running final evaluation...")
        logger.info("=" * 80)

        # load best model for evaluation
        if args.phase == '1':
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'phase1_best_model.pth')
        elif args.phase == '2':
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'phase2_best_model.pth')
        elif args.phase == '3':
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'phase3_best_model.pth')
        elif args.phase == '4':
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'phase4_best_model.pth')
        else:  # 'all' - use Phase 4 best model (final phase)
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'phase4_best_model.pth')

        if os.path.exists(best_checkpoint_path):
            logger.info(f"Loading best model from: {best_checkpoint_path}")
            checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"  Phase: {checkpoint.get('phase', 'unknown')}")
            logger.info(f"  Best validation accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")
        else:
            logger.warning(f"Best checkpoint not found at {best_checkpoint_path}")
            logger.warning("Using current model state for evaluation")

        # run final evaluation
        results = run_final_evaluation(model, dataloaders, device, args)

        # ========== TRAINING SUMMARY ==========
        logger.info("=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Phase: {args.phase}")
        logger.info(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
        logger.info(f"Final test accuracy: {results['accuracy']:.2f}%")
        if 'top5_accuracy' in results:
            logger.info(f"Final test top-5 accuracy: {results['top5_accuracy']:.2f}%")
        logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")
        logger.info("=" * 80)

        logger.info("Training and evaluation complete!")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user!")
        logger.info("Saving emergency checkpoint...")

        emergency_path = os.path.join(args.checkpoint_dir, 'emergency_checkpoint.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'interrupted': True
        }, emergency_path)

        logger.info(f"Emergency checkpoint saved to: {emergency_path}")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()