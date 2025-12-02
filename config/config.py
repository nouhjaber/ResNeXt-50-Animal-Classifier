import torch
import os

# ============================================================================
# PART 1: MODEL ARCHITECTURE CONFIGURATION
# ============================================================================

class ModelConfig:
    """
    ResNeXt-50 (32*4d) for AwA2 animal classification
    """
    
    # Input image size
    INPUT_HEIGHT = 224
    INPUT_WIDTH = 224
    INPUT_CHANNELS = 3  # RGB image
    
    # ResNeXt block configuration
    CARDINALITY = 32  # Number of parallel paths
    BASE_WIDTH = 4    # Width per path (32 * 4 = 128 channels bottleneck)
    
    #  50 animal classes
    NUM_CLASSES = 50
    
    # Activation function
    ACTIVATION = 'relu'
    
    # Batch normalization
    USE_BATCH_NORM = True
    BATCH_NORM_MOMENTUM = 0.1
    
    # Layer depth: ResNeXt-50 structure
    LAYER_BLOCKS = [3, 4, 6, 3]  # [Layer1, Layer2, Layer3, Layer4]
    
    # Channel progression
    INITIAL_CHANNELS = 64
    LAYER1_CHANNELS = 256
    LAYER2_CHANNELS = 512
    LAYER3_CHANNELS = 1024
    LAYER4_CHANNELS = 2048
    
    # Dropout
    DROPOUT_RATE = 0.5
    
    # Weight initialization
    INIT_METHOD = 'he'  # For ReLU activation


# ============================================================================
# PART 2: TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig:
    """
    4-Phase Progressive Fine-tuning Strategy for 92% Target Accuracy

    Phase 1: Layer4 + FC (15 epochs) - Target: 65-70%
    Phase 2: Layer3-4 + FC (25 epochs) - Target: 75-80%
    Phase 3: Layer2-4 + FC (25 epochs) - Target: 85-88%
    Phase 4: ALL layers (30 epochs) - Target: 90-92%

    Total: 95 epochs
    """

    # Phase 1: Layer4 + FC (freeze stem, layer1-3)
    PHASE1_EPOCHS = 15
    PHASE1_BATCH_SIZE = 32  # Tesla T4 can handle larger batches
    PHASE1_OPTIMIZER = 'adamw'
    PHASE1_LR_LAYER4 = 8e-5
    PHASE1_LR_FC = 3e-4

    # Phase 2: Layer3-4 + FC (freeze stem, layer1-2)
    PHASE2_EPOCHS = 25
    PHASE2_BATCH_SIZE = 32
    PHASE2_OPTIMIZER = 'adamw'
    PHASE2_LR_LAYER3 = 3e-5
    PHASE2_LR_LAYER4 = 8e-5
    PHASE2_LR_FC = 3e-4

    # Phase 3: Layer2-4 + FC (freeze stem, layer1)
    PHASE3_EPOCHS = 25
    PHASE3_BATCH_SIZE = 28  # Slightly smaller for more layers
    PHASE3_OPTIMIZER = 'adamw'
    PHASE3_LR_LAYER2 = 1e-5
    PHASE3_LR_LAYER3 = 3e-5
    PHASE3_LR_LAYER4 = 8e-5
    PHASE3_LR_FC = 3e-4

    # Phase 4: ALL layers trainable
    PHASE4_EPOCHS = 30
    PHASE4_BATCH_SIZE = 24  # Smaller batch for all layers
    PHASE4_OPTIMIZER = 'adamw'
    PHASE4_LR_STEM = 1e-6      # Very very low
    PHASE4_LR_LAYER1 = 5e-6    # Extremely low
    PHASE4_LR_LAYER2 = 1e-5    # Very low
    PHASE4_LR_LAYER3 = 3e-5    # Low
    PHASE4_LR_LAYER4 = 8e-5    # Medium
    PHASE4_LR_FC = 3e-4        # High

    # Optimizer settings (AdamW for better regularization)
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4  # L2 regularization
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999

    # Loss function
    LOSS_FUNCTION = 'cross_entropy'  # Standard for 50-class classification

    # Learning rate scheduling - CosineAnnealingWarmRestarts
    USE_LR_SCHEDULER = True
    SCHEDULER_TYPE = 'cosine_warm_restarts'
    SCHEDULER_T0 = 10          # Initial restart period
    SCHEDULER_T_MULT = 2       # Multiply period by 2 after each restart
    SCHEDULER_ETA_MIN = 1e-7   # Minimum learning rate

    # Gradient clipping
    USE_GRADIENT_CLIPPING = True
    GRADIENT_CLIP_VALUE = 1.0
    GRADIENT_CLIP_NORM_TYPE = 2.0     # L2 norm clipping

    # Early stopping (disabled for robust training to completion)
    USE_EARLY_STOPPING = False  # Train full epochs for each phase
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_MIN_DELTA = 0.1    # Minimum improvement threshold (0.1%)

    # Checkpointing (UPDATED: Save best model only, keep for resume)
    SAVE_BEST_MODEL = True
    SAVE_LATEST_MODEL = True          # For resume capability after interruption
    SAVE_CHECKPOINT_INTERVAL = 999    # Don't save periodic checkpoints (save space)
    SAVE_LAST_N_CHECKPOINTS = 1       # Only keep best model


# ============================================================================
# PART 3: DATA CONFIGURATION 
# ============================================================================

class DataConfig:
    """
    Data loading and preprocessing for AwA2 PNG images
    """
    
    # File paths
    DATA_ROOT = './data'  # Root directory
    RAW_DATA_PATH = './data/raw'  # Where AwA2 PNG files are stored
    PROCESSED_DATA_PATH = './data/processed'
    SPLITS_PATH = './data/splits'  # train/val/test split files
    
    IMAGE_FORMAT = 'png'
    SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg']
    
    # Use ImageNet normalization because model pre-trained on ImageNet
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # RGB channels
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Animals can appear at different angles, sizes, lighting
    USE_AUGMENTATION = True
    
    # Augmentation settings
    RANDOM_HORIZONTAL_FLIP = 0.5  # Animals can face either direction
    RANDOM_VERTICAL_FLIP = 0.0    # Don't flip upside down
    RANDOM_ROTATION = 10          # Animals at various angles
    RANDOM_CROP = (200, 200)      # Random crop for scale invariance
    RANDOM_BRIGHTNESS = 0.1       # Lighting variations
    RANDOM_CONTRAST = 0.1         # Contrast variations
    
    # Dataset splits
    TRAIN_SPLIT = 0.7  
    VAL_SPLIT = 0.15   
    TEST_SPLIT = 0.15  
    
    # Data loader settings
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PREFETCH_FACTOR = 4
    SHUFFLE_TRAIN = True
    SHUFFLE_VAL = False
    SHUFFLE_TEST = False
    
    # Class weights (handle if some animals have fewer images)
    USE_CLASS_WEIGHTS = True
    
    # Data validation
    VALIDATE_DATA = True


# ============================================================================
# PART 4: DEVICE CONFIGURATION
# ============================================================================

class DeviceConfig:
    """
    GPU/CPU settings - Optimized for Tesla T4 15GB
    """

    # Automatic device selection
    USE_CUDA = torch.cuda.is_available()
    DEVICE = 'cuda' if USE_CUDA else 'cpu'

    # GPU settings
    GPU_ID = 0

    # Memory optimization for Tesla T4 15GB
    EMPTY_CACHE_INTERVAL = 50  # More frequent cache clearing for stability
    USE_AMP = True  # Tesla T4 OPTIMIZATION: Turing architecture with Tensor Cores (2x speedup)

    # Tesla T4 specific optimizations
    CUDNN_BENCHMARK = True      # Auto-tune conv algorithms for T4
    CUDNN_DETERMINISTIC = False # Disable for better performance (enable if reproducibility critical)

    # Reproducibility
    SEED = 42
    DETERMINISTIC = False  # Disabled for better T4 performance

    # Precision
    MODEL_DTYPE = torch.float32

    # Multi-GPU (Tesla T4 typically single GPU in Colab)
    USE_MULTI_GPU = torch.cuda.device_count() > 1
    NUM_GPUS = torch.cuda.device_count()

    # Memory estimates for Tesla T4 15GB
    ESTIMATED_MODEL_SIZE_GB = 0.5   # ResNeXt-50 model size
    ESTIMATED_BATCH_MEMORY_GB = {
        'batch_16': 3.0,
        'batch_24': 4.5,
        'batch_28': 5.2,
        'batch_32': 6.0
    }


# ============================================================================
# PART 5: EVALUATION CONFIGURATION
# ============================================================================

class EvalConfig:
    # Validation frequency
    VALIDATE_EVERY_N_BATCHES = 100

    # Metrics to compute
    COMPUTE_METRICS = ['accuracy', 'precision', 'recall', 'f1']

    # Confusion matrix (NO PNG saving, only compute)
    SAVE_CONFUSION_MATRIX = False  # Don't save PNG files
    COMPUTE_CONFUSION_MATRIX = True  # But compute for analysis

    # Visualization (DISABLED - no PNG saving)
    VISUALIZE_PREDICTIONS = False
    SAVE_VISUALIZATIONS = False
    NUM_VISUALIZATIONS = 0

    # Confidence threshold
    CONFIDENCE_THRESHOLD = 0.5


# ============================================================================
# PART 6: TRANSFER LEARNING CONFIGURATION
# ============================================================================

class TransferLearningConfig:
    """
    Which layers stay frozen (ImageNet knowledge) vs trainable (AwA2 task)
    """
    
    # Use ImageNet pre-trained weights
    USE_PRETRAINED = True
    PRETRAINED_SOURCE = 'torchvision'
    
    # Layer freezing strategy
    FREEZE_STEM = True      # Freeze basic image feature detection
    FREEZE_LAYER1 = True    # Freeze simple feature extraction
    FREEZE_LAYER2 = True    # Freeze mid-level features
    FREEZE_LAYER3 = True    # Freeze complex features
    FREEZE_LAYER4 = False   # UNFREEZE for animal-specific fine-tuning
    FREEZE_FC = False       # UNFREEZE classification head


# ============================================================================
# PART 7: LOGGING AND TRACKING CONFIGURATION
# ============================================================================

class LoggingConfig:
    """
    Experiment tracking and checkpoint saving - Minimal logging for Colab
    """

    # Directories
    LOGS_DIR = './logs'
    CHECKPOINT_DIR = './models/checkpoints'
    RESULTS_DIR = './results'

    # Experiment name
    EXPERIMENT_NAME = 'resnext50_awa2_4phase_v1'

    # TensorBoard logging (DISABLED - not needed for Colab)
    USE_TENSORBOARD = False

    # File logging (DISABLED - console only for Colab)
    USE_FILE_LOGGING = False
    LOG_FILE_PATH = None

    # Logging settings (console only)
    LOG_INTERVAL = 10  # Log every 10 batches
    LOG_LOSS = True
    LOG_ACCURACY = True
    LOG_LEARNING_RATE = True
    LOG_GRADIENTS = False
    LOG_WEIGHTS = False

    # Console output
    VERBOSE = True
    PRINT_INTERVAL = 10

    # Save predictions (DISABLED - no need to save prediction files)
    SAVE_PREDICTIONS = False
    SAVE_PRED_INTERVAL = None

    # PNG/Image saving (DISABLED - no visualization images)
    SAVE_PLOTS = False
    SAVE_CONFUSION_MATRIX_PNG = False
    SAVE_TRAINING_CURVES_PNG = False


# ============================================================================
# PART 8: INFERENCE CONFIGURATION
# ============================================================================

class InferenceConfig:
    # Model checkpoint
    MODEL_CHECKPOINT_PATH = './models/checkpoints/best_model.pth'
    
    # Prediction settings
    CONFIDENCE_THRESHOLD = 0.7
    ENSEMBLE_SIZE = 1
    
    # Output
    RETURN_PROBABILITIES = True  # Get confidence for each animal class
    RETURN_ATTENTION = False
    
    # Batch inference
    INFERENCE_BATCH_SIZE = 32


# ============================================================================
# HELPER FUNCTION
# ============================================================================

def get_phase_config(phase):
    """
    Get configuration for each training phase (4-phase strategy)
    """
    if phase == 1:
        return {
            'epochs': TrainingConfig.PHASE1_EPOCHS,
            'batch_size': TrainingConfig.PHASE1_BATCH_SIZE,
            'optimizer': TrainingConfig.PHASE1_OPTIMIZER,
            'freeze_layers': ['stem', 'layer1', 'layer2', 'layer3'],
            'trainable_layers': ['layer4', 'fc'],
            'learning_rates': {
                'layer4': TrainingConfig.PHASE1_LR_LAYER4,
                'fc': TrainingConfig.PHASE1_LR_FC
            },
            'description': 'Phase 1: Layer4 + FC (15 epochs) - Target: 65-70%',
            'target_accuracy': '65-70%'
        }
    elif phase == 2:
        return {
            'epochs': TrainingConfig.PHASE2_EPOCHS,
            'batch_size': TrainingConfig.PHASE2_BATCH_SIZE,
            'optimizer': TrainingConfig.PHASE2_OPTIMIZER,
            'freeze_layers': ['stem', 'layer1', 'layer2'],
            'trainable_layers': ['layer3', 'layer4', 'fc'],
            'learning_rates': {
                'layer3': TrainingConfig.PHASE2_LR_LAYER3,
                'layer4': TrainingConfig.PHASE2_LR_LAYER4,
                'fc': TrainingConfig.PHASE2_LR_FC
            },
            'description': 'Phase 2: Layer3-4 + FC (25 epochs) - Target: 75-80%',
            'target_accuracy': '75-80%'
        }
    elif phase == 3:
        return {
            'epochs': TrainingConfig.PHASE3_EPOCHS,
            'batch_size': TrainingConfig.PHASE3_BATCH_SIZE,
            'optimizer': TrainingConfig.PHASE3_OPTIMIZER,
            'freeze_layers': ['stem', 'layer1'],
            'trainable_layers': ['layer2', 'layer3', 'layer4', 'fc'],
            'learning_rates': {
                'layer2': TrainingConfig.PHASE3_LR_LAYER2,
                'layer3': TrainingConfig.PHASE3_LR_LAYER3,
                'layer4': TrainingConfig.PHASE3_LR_LAYER4,
                'fc': TrainingConfig.PHASE3_LR_FC
            },
            'description': 'Phase 3: Layer2-4 + FC (25 epochs) - Target: 85-88%',
            'target_accuracy': '85-88%'
        }
    elif phase == 4:
        return {
            'epochs': TrainingConfig.PHASE4_EPOCHS,
            'batch_size': TrainingConfig.PHASE4_BATCH_SIZE,
            'optimizer': TrainingConfig.PHASE4_OPTIMIZER,
            'freeze_layers': [],  # No frozen layers
            'trainable_layers': ['stem', 'layer1', 'layer2', 'layer3', 'layer4', 'fc'],
            'learning_rates': {
                'stem': TrainingConfig.PHASE4_LR_STEM,
                'layer1': TrainingConfig.PHASE4_LR_LAYER1,
                'layer2': TrainingConfig.PHASE4_LR_LAYER2,
                'layer3': TrainingConfig.PHASE4_LR_LAYER3,
                'layer4': TrainingConfig.PHASE4_LR_LAYER4,
                'fc': TrainingConfig.PHASE4_LR_FC
            },
            'description': 'Phase 4: ALL layers (30 epochs) - Target: 90-92%',
            'target_accuracy': '90-92%'
        }
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 1, 2, 3, or 4")
