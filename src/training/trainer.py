import os
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.config import TrainingConfig, LoggingConfig, DeviceConfig

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# TRAINER CLASS
# ============================================================================

class Trainer:
    """
    ResNeXt-50 on AwA2 dataset

    Features:
     Two-phase training (Phase 1 FC only, Phase 2 Layer4 + FC)
     Automatic checkpointing (save best model based on val accuracy)
     TensorBoard logging
     Early stopping
     Learning rate scheduling
     Gradient clipping
     Comprehensive metrics tracking
    """

    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        config: Optional[object] = None
    ):
        # Model and device
        self.model = model
        self.device = device if device is not None else torch.device(DeviceConfig.DEVICE)
        self.model.to(self.device)

        # Data
        self.dataloaders = dataloaders
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']

        # optimization
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # configuration
        self.config = config if config is not None else TrainingConfig

        # Gradient clipping
        self.use_grad_clip = TrainingConfig.USE_GRADIENT_CLIPPING
        self.grad_clip_value = TrainingConfig.GRADIENT_CLIP_VALUE

        # Early stopping
        self.use_early_stopping = TrainingConfig.USE_EARLY_STOPPING
        self.early_stopping_patience = TrainingConfig.EARLY_STOPPING_PATIENCE
        self.early_stopping_counter = 0

        # tracking
        self.current_epoch = 0
        self.current_phase = 1  # Track which phase we're in
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        # Time tracking for ETA and estimation
        self.epoch_times = []
        self.total_epochs = 0
        self.phase_start_time = None
        self.total_training_start_time = None

        # Logging
        self.use_tensorboard = LoggingConfig.USE_TENSORBOARD
        if self.use_tensorboard:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = os.path.join(
                LoggingConfig.TENSORBOARD_DIR,
                f"{LoggingConfig.EXPERIMENT_NAME}_{timestamp}"
            )
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging to: {log_dir}")
        else:
            self.writer = None

        # Checkpoints
        self.checkpoint_dir = LoggingConfig.CHECKPOINT_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # MX450 OPTIMIZATION: Mixed Precision Training (AMP)
        self.use_amp = DeviceConfig.USE_AMP and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info(f"Mixed Precision (AMP) ENABLED - Using Tensor Cores for 2x speedup")
        else:
            self.scaler = None
            logger.info(f"Mixed Precision (AMP) disabled")

        logger.info(f"Trainer initialized on device: {self.device}")

    def _train_epoch(self) -> Tuple[float, float]:
        # Set model to train mode (but frozen layers stay in eval mode)
        self.model.train()

        # Re apply eval mode to frozen layers if they exist
        if hasattr(self.model, 'stem') and not any(p.requires_grad for p in self.model.stem.parameters()):
            self.model.stem.eval()
        if hasattr(self.model, 'layer1') and not any(p.requires_grad for p in self.model.layer1.parameters()):
            self.model.layer1.eval()
        if hasattr(self.model, 'layer2') and not any(p.requires_grad for p in self.model.layer2.parameters()):
            self.model.layer2.eval()
        if hasattr(self.model, 'layer3') and not any(p.requires_grad for p in self.model.layer3.parameters()):
            self.model.layer3.eval()
        if hasattr(self.model, 'layer4') and not any(p.requires_grad for p in self.model.layer4.parameters()):
            self.model.layer4.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar with detailed info
        pbar = tqdm(
            self.train_loader,
            desc=f"Phase {self.current_phase} | Epoch {self.current_epoch + 1}/{self.total_epochs} [Train]",
            leave=False,
            disable=False,
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            # move to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # zero gradients
            self.optimizer.zero_grad()

            # MX450 OPTIMIZATION: Mixed precision forward/backward pass
            if self.use_amp:
                # Forward pass with autocast (uses FP16 for Tensor Core operations)
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping (unscale first for correct norm calculation)
                if self.use_grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_value
                    )

                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard FP32 training
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # gradient clipping
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_value
                    )

                # Optimizer step
                self.optimizer.step()

            # Step CosineAnnealingWarmRestarts scheduler every batch
            if self.scheduler is not None and isinstance(
                self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
            ):
                self.scheduler.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            current_loss = running_loss / total
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        # Calculate epoch metrics
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def _validate_epoch(self) -> Tuple[float, float]:
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar with detailed info
        pbar = tqdm(
            self.val_loader,
            desc=f"Phase {self.current_phase} | Epoch {self.current_epoch + 1}/{self.total_epochs} [Val]",
            leave=False,
            disable=False,
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        with torch.no_grad():
            for images, labels in pbar:
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # MX450 OPTIMIZATION Use AMP for faster inference
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    # Standard FP32 inference
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                current_loss = running_loss / total
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%'
                })

        # Calculate epoch metrics
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return epoch_loss, epoch_acc

    def _save_checkpoint(
        self,
        epoch: int,
        val_acc: float,
        val_loss: float,
        is_best: bool = False
    ):
        # Save model checkpoint with complete training state for resume
        checkpoint = {
            'epoch': epoch,
            'phase': self.current_phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'early_stopping_counter': self.early_stopping_counter
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if TrainingConfig.SAVE_LATEST_MODEL:
            latest_path = os.path.join(
                self.checkpoint_dir,
                f'phase{self.current_phase}_latest_checkpoint.pth'
            )
            torch.save(checkpoint, latest_path)

        # Save best checkpoint only when validation improves
        if is_best and TrainingConfig.SAVE_BEST_MODEL:
            best_path = os.path.join(
                self.checkpoint_dir,
                f'phase{self.current_phase}_best_model.pth'
            )
            torch.save(checkpoint, best_path)
            logger.info(f"✓ Phase {self.current_phase} - Best model saved (val_acc: {val_acc:.2f}%)")


    def _log_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float
    ):

        # TensorBoard logging
        if self.writer is not None:
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', lr, epoch)
            self.writer.flush() 

        # Console logging
        logger.info("=" * 80)
        logger.info(f"Epoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        logger.info(f"  Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch + 1})")
        logger.info(f"  Learning Rate: {lr:.6f}")
        logger.info("=" * 80)

    def _check_early_stopping(self, val_acc: float) -> bool:
        if not self.use_early_stopping:
            return False

        if val_acc > self.best_val_acc:
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            logger.info(
                f"Early stopping counter: {self.early_stopping_counter}/"
                f"{self.early_stopping_patience}"
            )

            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered! No improvement for "
                    f"{self.early_stopping_patience} epochs."
                )
                return True

        return False

    def _get_current_lr(self) -> float:
        # get current learning rate from optimizer
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def _estimate_training_time(self, phase: int, avg_epoch_time: float = None):
        """
        Estimate training time for phases
        Returns dict with time estimates
        """
        from config.config import TrainingConfig

        # Use average epoch time if available, otherwise estimate based on batch size
        if avg_epoch_time is None:
            # Rough estimate: ~2-3 minutes per epoch for batch_size=32 on Tesla T4
            avg_epoch_time = 120  # seconds (conservative estimate)

        phase_epochs = {
            1: TrainingConfig.PHASE1_EPOCHS,
            2: TrainingConfig.PHASE2_EPOCHS,
            3: TrainingConfig.PHASE3_EPOCHS,
            4: TrainingConfig.PHASE4_EPOCHS
        }

        estimates = {}
        for p in range(1, 5):
            estimates[f'phase{p}'] = phase_epochs[p] * avg_epoch_time

        # Total time
        estimates['total'] = sum(estimates[f'phase{i}'] for i in range(1, 5))

        # Remaining time (from current phase onwards)
        estimates['remaining'] = sum(estimates[f'phase{i}'] for i in range(phase, 5))

        return estimates

    def train(self, num_epochs: int, phase: int = 1, resume: bool = False) -> Dict:
        # Main training loop

        # Set current phase for checkpoint naming
        self.current_phase = phase
        self.total_epochs = num_epochs

        start_epoch = self.current_epoch + 1 if resume else 0

        # Track phase start time
        self.phase_start_time = time.time()

        logger.info("=" * 80)
        if resume:
            logger.info(f"Resuming Training - Phase {phase}")
            logger.info(f"Starting from epoch {start_epoch + 1}/{num_epochs}")
        else:
            logger.info(f"Starting Training - Phase {phase}")
            logger.info(f"Total Epochs: {num_epochs}")
        logger.info(f"Device: {self.device}")

        # Print model info
        if hasattr(self.model, 'get_trainable_params'):
            trainable, total = self.model.get_trainable_params()
            logger.info(f"Trainable parameters: {trainable:,} / {total:,}")

        if not resume and len(self.epoch_times) > 0:
            # Use actual average from previous training
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        else:
            # Use estimate for first run
            avg_epoch_time = None

        time_estimates = self._estimate_training_time(phase, avg_epoch_time)

        logger.info("")
        logger.info("TIME ESTIMATES (Tesla T4 15GB):")
        logger.info(f"  Current phase ({num_epochs} epochs): ~{self._format_time(num_epochs * (avg_epoch_time or 120))}")
        logger.info(f"  Phase 1 (15 epochs): ~{self._format_time(time_estimates['phase1'])}")
        logger.info(f"  Phase 2 (25 epochs): ~{self._format_time(time_estimates['phase2'])}")
        logger.info(f"  Phase 3 (25 epochs): ~{self._format_time(time_estimates['phase3'])}")
        logger.info(f"  Phase 4 (30 epochs): ~{self._format_time(time_estimates['phase4'])}")
        logger.info(f"  TOTAL (95 epochs): ~{self._format_time(time_estimates['total'])}")
        logger.info(f"  Remaining (from Phase {phase}): ~{self._format_time(time_estimates['remaining'])}")

        logger.info("=" * 80)

        start_time = time.time()

        # Reset epoch times for fresh ETA calculation
        if not resume:
            self.epoch_times = []

        warmup_epochs = 3 if phase == 1 else 5
        base_lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        logger.info(f"Using {warmup_epochs} warmup epochs for smoother convergence")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            if epoch < warmup_epochs and not resume:
                warmup_factor = (epoch + 1) / warmup_epochs
                for idx, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = base_lrs[idx] * warmup_factor
                logger.info(f"Warmup: LR scaled by {warmup_factor:.3f}")
            elif epoch == warmup_epochs and not resume:
                # Restore base learning rates after warmup
                for idx, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = base_lrs[idx]
                logger.info("Warmup complete - using base learning rates")

            # Train one epoch
            train_loss, train_acc = self._train_epoch()

            # Validate one epoch
            val_loss, val_acc = self._validate_epoch()

            current_lr = self._get_current_lr()

            # update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            # early stopping check - MUST BE BEFORE updating best_val_acc
            if self._check_early_stopping(val_acc):
                # Still need to check if this epoch is best and save before breaking
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch

                if TrainingConfig.SAVE_BEST_MODEL:
                    self._save_checkpoint(epoch, val_acc, val_loss, is_best)

                self._log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)

                # Calculate timing info even for early stopping
                epoch_time = time.time() - epoch_start_time
                self.epoch_times.append(epoch_time)

                logger.info(f"Stopping training at epoch {epoch + 1}")
                logger.info(f"Epoch time: {self._format_time(epoch_time)}")
                break

            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch

            # save checkpoint
            if TrainingConfig.SAVE_BEST_MODEL:
                self._save_checkpoint(epoch, val_acc, val_loss, is_best)

            self._log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(-val_acc)
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    pass
                else:
                    self.scheduler.step()

            # Epoch timing and ETA calculation
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)

            # Calculate ETA based on average epoch time
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            epochs_remaining = num_epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * epochs_remaining

            # Progress percentage
            progress_pct = ((epoch + 1) / num_epochs) * 100

            # Format time displays
            epoch_time_str = self._format_time(epoch_time)
            avg_time_str = self._format_time(avg_epoch_time)
            eta_str = self._format_time(eta_seconds)

            # Create progress bar visualization
            bar_length = 30
            filled_length = int(bar_length * (epoch + 1) // num_epochs)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            logger.info(f"Progress: [{bar}] {progress_pct:.1f}% ({epoch + 1}/{num_epochs} epochs)")
            logger.info(f"Epoch: {epoch_time_str} | Avg: {avg_time_str}/epoch | ETA: {eta_str}\n")

        # Training complete
        total_time = time.time() - start_time
        phase_time = time.time() - self.phase_start_time if self.phase_start_time else total_time
        total_time_str = self._format_time(total_time)
        phase_time_str = self._format_time(phase_time)

        logger.info("=" * 80)
        logger.info(f"Phase {self.current_phase} Training Complete!")
        logger.info(f"Phase {self.current_phase} time: {phase_time_str} ({phase_time:.0f}s)")
        logger.info(f"Total epochs completed: {epoch + 1}/{num_epochs}")
        if self.epoch_times:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            logger.info(f"Average time per epoch: {self._format_time(avg_epoch_time)} ({avg_epoch_time:.0f}s)")
            logger.info(f"Estimated time per epoch: ~{self._format_time(avg_epoch_time)}")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch + 1})")
        logger.info("=" * 80)

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()

        return self.history
    
    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = False):
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
        # GPU compatible checkpoint loading
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
    
        # always load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
        if resume_training:
            # Restore full training state with error handling
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Optimizer state restored successfully")
            except (ValueError, KeyError) as e:
                if "parameter groups" in str(e) or "state" in str(e):
                    logger.warning(f"Optimizer state mismatch: {e}")
                    logger.warning("Starting with fresh optimizer state (parameter groups changed)")
                    # Continue without optimizer state - will use fresh optimizer
                else:
                    logger.error(f"Failed to load optimizer state: {e}")
                    raise e
    
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info("Scheduler state restored")
                except (ValueError, KeyError) as e:
                    logger.warning(f"Scheduler state mismatch: {e}")
                    logger.warning("Starting with fresh scheduler state")
                    # Continue without scheduler state
    
            # MX450 OPTIMIZATION: Restore AMP scaler state
            if self.use_amp and self.scaler is not None and 'scaler_state_dict' in checkpoint:
                try:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    logger.info("AMP scaler state restored")
                except (ValueError, KeyError) as e:
                    logger.warning(f"AMP scaler state mismatch: {e}")
                    logger.warning("Starting with fresh AMP scaler state")
    
            self.current_epoch = checkpoint.get('epoch', 0)
            self.current_phase = checkpoint.get('phase', 1)
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            self.best_epoch = checkpoint.get('best_epoch', 0)
            self.history = checkpoint.get('history', self.history)
            self.early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
    
            logger.info(f"Training state restored from epoch {self.current_epoch + 1}")
            logger.info(f"Phase: {self.current_phase}")
            logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch + 1})")
            logger.info(f"Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
        else:
            # Only load weights for evaluation
            logger.info(f"Model weights loaded (evaluation mode)")
            if 'phase' in checkpoint:
                logger.info(f"Checkpoint phase: {checkpoint['phase']}")
            if 'best_val_acc' in checkpoint:
                logger.info(f"Checkpoint validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    def evaluate(self, dataloader: Optional[torch.utils.data.DataLoader] = None) -> Dict:

        if dataloader is None:
            dataloader = self.val_loader

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        logger.info("Running evaluation...")

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # calculate metrics
        avg_loss = running_loss / total
        accuracy = 100. * correct / total

        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }

        logger.info(f"Evaluation Results:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Accuracy: {accuracy:.2f}%")

        return results


# ============================================================================
# TRAINER FACTORY FUNCTION
# ============================================================================

def create_trainer(
    model: nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    phase: int = 1,
    device: Optional[torch.device] = None
) -> Trainer:
    """
    Trainer factory for 4-phase transfer learning

    Phase 1 -> Layer4 + FC (15 epochs) - Target: 65-70%
    Phase 2 -> Layer3-4 + FC (25 epochs) - Target: 75-80%
    Phase 3 -> Layer2-4 + FC (25 epochs) - Target: 85-88%
    Phase 4 -> ALL layers (30 epochs) - Target: 90-92%
    """

    if device is None:
        device = torch.device(DeviceConfig.DEVICE)

    from config.config import get_phase_config
    phase_cfg = get_phase_config(phase)

    logger.info("=" * 80)
    logger.info(f"{phase_cfg['description']}")
    logger.info("=" * 80)

    # Apply phase-specific freezing
    freeze_method = f'freeze_for_phase{phase}'
    if hasattr(model, freeze_method):
        getattr(model, freeze_method)()
    else:
        raise ValueError(f"Model missing {freeze_method} method")

    # ========================================================================
    # PHASE-SPECIFIC OPTIMIZER WITH LAYERWISE LEARNING RATES
    # ========================================================================
    param_groups = []

    # Phase 1: Layer4 + FC
    if phase == 1:
        param_groups.append({
            'params': [p for p in model.layer4.parameters() if p.requires_grad],
            'lr': TrainingConfig.PHASE1_LR_LAYER4,
            'name': 'layer4'
        })
        param_groups.append({
            'params': [p for p in model.fc.parameters() if p.requires_grad],
            'lr': TrainingConfig.PHASE1_LR_FC,
            'name': 'fc'
        })

    # Phase 2: Layer3-4 + FC
    elif phase == 2:
        param_groups.append({
            'params': [p for p in model.layer3.parameters() if p.requires_grad],
            'lr': TrainingConfig.PHASE2_LR_LAYER3,
            'name': 'layer3'
        })
        param_groups.append({
            'params': [p for p in model.layer4.parameters() if p.requires_grad],
            'lr': TrainingConfig.PHASE2_LR_LAYER4,
            'name': 'layer4'
        })
        param_groups.append({
            'params': [p for p in model.fc.parameters() if p.requires_grad],
            'lr': TrainingConfig.PHASE2_LR_FC,
            'name': 'fc'
        })

    # Phase 3: Layer2-4 + FC
    elif phase == 3:
        param_groups.append({
            'params': [p for p in model.layer2.parameters() if p.requires_grad],
            'lr': TrainingConfig.PHASE3_LR_LAYER2,
            'name': 'layer2'
        })
        param_groups.append({
            'params': [p for p in model.layer3.parameters() if p.requires_grad],
            'lr': TrainingConfig.PHASE3_LR_LAYER3,
            'name': 'layer3'
        })
        param_groups.append({
            'params': [p for p in model.layer4.parameters() if p.requires_grad],
            'lr': TrainingConfig.PHASE3_LR_LAYER4,
            'name': 'layer4'
        })
        param_groups.append({
            'params': [p for p in model.fc.parameters() if p.requires_grad],
            'lr': TrainingConfig.PHASE3_LR_FC,
            'name': 'fc'
        })

    # Phase 4: ALL layers
    elif phase == 4:
        param_groups.append({
            'params': [p for p in model.stem.parameters() if p.requires_grad],
            'lr': TrainingConfig.PHASE4_LR_STEM,
            'name': 'stem'
        })
        param_groups.append({
            'params': [p for p in model.layer1.parameters() if p.requires_grad],
            'lr': TrainingConfig.PHASE4_LR_LAYER1,
            'name': 'layer1'
        })
        param_groups.append({
            'params': [p for p in model.layer2.parameters() if p.requires_grad],
            'lr': TrainingConfig.PHASE4_LR_LAYER2,
            'name': 'layer2'
        })
        param_groups.append({
            'params': [p for p in model.layer3.parameters() if p.requires_grad],
            'lr': TrainingConfig.PHASE4_LR_LAYER3,
            'name': 'layer3'
        })
        param_groups.append({
            'params': [p for p in model.layer4.parameters() if p.requires_grad],
            'lr': TrainingConfig.PHASE4_LR_LAYER4,
            'name': 'layer4'
        })
        param_groups.append({
            'params': [p for p in model.fc.parameters() if p.requires_grad],
            'lr': TrainingConfig.PHASE4_LR_FC,
            'name': 'fc'
        })

    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 1, 2, 3, or 4")

    # Create AdamW optimizer with layerwise learning rates
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=TrainingConfig.WEIGHT_DECAY,
        betas=(TrainingConfig.ADAM_BETA1, TrainingConfig.ADAM_BETA2)
    )

    # CosineAnnealingWarmRestarts scheduler
    if TrainingConfig.USE_LR_SCHEDULER and TrainingConfig.SCHEDULER_TYPE == 'cosine_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=TrainingConfig.SCHEDULER_T0,
            T_mult=TrainingConfig.SCHEDULER_T_MULT,
            eta_min=TrainingConfig.SCHEDULER_ETA_MIN
        )
        scheduler_name = f"CosineAnnealingWarmRestarts (T_0={TrainingConfig.SCHEDULER_T0}, T_mult={TrainingConfig.SCHEDULER_T_MULT})"
    else:
        scheduler = None
        scheduler_name = "None"

    # Log configuration
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Optimizer: AdamW")
    logger.info(f"Learning rates per layer:")
    for pg in param_groups:
        logger.info(f"  {pg['name']}: {pg['lr']:.6f}")
    logger.info(f"Weight decay: {TrainingConfig.WEIGHT_DECAY}")
    logger.info(f"Scheduler: {scheduler_name}")
    logger.info("=" * 80)

    # Create trainer
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    return trainer