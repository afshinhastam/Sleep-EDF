# Sleep-EDF EEG Classification Project

A deep learning framework for automatic sleep stage classification using EEG data from the Sleep-EDF database. This project implements both object-based and subject-based training approaches with a sophisticated CNN-based architecture featuring attention mechanisms.

## 🧠 Project Overview

This project provides a complete pipeline for:
- **EEG Data Preprocessing**: Converting raw EDF files to windowed epochs
- **Deep Learning Model**: CNN-based architecture with feature attention for sleep stage classification
- **Two Training Paradigms**: Object-based and subject-based approaches
- **Comprehensive Evaluation**: Training visualization, hypnogram plotting, and performance metrics

## 📁 Project Structure

```
Sleep-edf/
├── Model/
│   └── Model.py              # Main EEG classification model architecture
├── ObjectBased/
│   └── Train.py              # Object-based training pipeline
├── SubjectBased/
│   └── Train.py              # Subject-based training pipeline
└── Utils/
    ├── PreProcessing.py      # EEG data preprocessing utilities
    └── Utils.py              # Dataset classes and utility functions
```

## 🏗️ Model Architecture

The `EEGClassifierModel` consists of three main components:

### 1. Backbone (`BackBone`)
- **TimeDistributed CNN**: Applies CNN blocks across time steps
- **CNNBlock**: 5-layer convolutional network with batch normalization, ReLU, and dropout
- **FeatureAttention**: Multi-head attention mechanism for feature refinement

### 2. Bottleneck (`BottleNeck`)
- Dilated convolutions for capturing long-range dependencies
- Progressive dilation rates (1, 2, 4, 8) for multi-scale feature extraction

### 3. Classification Head (`Head`)
- Fully connected layers with batch normalization
- Softmax activation for 5-class sleep stage classification

**Sleep Stages**: N1, N2, N3, Wake (W), REM (R)

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install mne numpy pandas matplotlib tqdm tensorboard
```

### Data Preparation

1. **Download Sleep-EDF Database**: Place your EDF files in the appropriate directory structure
2. **Preprocess Data**: Convert EDF files to windowed epochs

```bash
python Utils/PreProcessing.py \
    --epoch_size 29.99 \
    --save_epochs_dir epochs \
    --root_edf_path "path/to/your/edf/files" \
    --save_seq_path seq_epochs \
    --seq_len 10 \
    --seq_shift 5
```

### Training

#### Object-Based Training
Trains on randomly mixed epochs from all subjects:

```bash
python ObjectBased/Train.py \
    --batch_size 48 \
    --n_epochs 10 \
    --lr 5e-4 \
    --root_path "seq_epochs" \
    --save_dir "checkpoints/object_based"
```

#### Subject-Based Training
Trains with subject-aware batching (keeps subjects together):

```bash
python SubjectBased/Train.py \
    --batch_size 48 \
    --n_epochs 10 \
    --lr 5e-4 \
    --root_path "seq_epochs" \
    --info_path "seq_epochs/info_csv.csv" \
    --save_dir "checkpoints/subject_based"
```

## 📊 Key Features

### Training Features
- **TensorBoard Integration**: Real-time training metrics and visualization
- **Checkpointing**: Automatic model saving with epoch tracking
- **Resume Training**: Continue from any saved checkpoint
- **Progress Bars**: Real-time training progress with tqdm
- **GPU/CPU Support**: Automatic device detection and utilization

### Data Handling
- **Sequence Windows**: Configurable sequence length and shift for temporal modeling
- **Subject-Aware Batching**: Maintains subject integrity in batches
- **Memory Efficient**: Lazy loading for large datasets
- **Data Augmentation**: Built-in preprocessing pipeline

### Evaluation
- **Hypnogram Visualization**: Compare predicted vs. ground truth sleep stages
- **Performance Metrics**: Accuracy tracking across train/validation/test sets
- **Training Plots**: Loss and accuracy curves
- **Class Distribution**: Per-subject and global class statistics

## 🔧 Configuration

### Model Parameters
- **Input Shape**: `(batch_size, sequence_length, channels, time_points)`
- **Default Sequence Length**: 10 epochs (30 seconds each)
- **Number of Classes**: 5 sleep stages
- **Attention Heads**: 3 multi-head attention

### Training Parameters
- **Default Batch Size**: 48
- **Default Learning Rate**: 5e-4
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Data Split**: 70% train, 20% validation, 10% test

## 📈 Results and Visualization

The training pipeline automatically generates:
- Training loss and accuracy plots
- TensorBoard logs for detailed monitoring
- Hypnogram comparisons
- Model checkpoints for each epoch

## 🛠️ Customization

### Adding New Models
Extend the `Model.py` file to add new architectures while maintaining the same interface.

### Custom Datasets
Implement new dataset classes inheriting from `torch.utils.data.Dataset` in `Utils.py`.

### Hyperparameter Tuning
Modify the argument parser in training scripts to experiment with different configurations.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{sleep_edf_classification,
  title={Sleep-EDF EEG Classification with Deep Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/sleep-edf}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- New model architectures
- Performance improvements
- Documentation enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Sleep-EDF Database for providing the EEG data
- PyTorch team for the deep learning framework
- MNE-Python for EEG data processing capabilities
- The open-source community for various utility libraries

---

**Note**: This project is designed for research purposes. For clinical applications, additional validation and regulatory compliance may be required. 