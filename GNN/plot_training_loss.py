import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import glob

def find_log_files(model_dir):
    """
    Find all log files for a model, including resume logs
    
    Returns:
    - List of log file paths sorted by creation/resume order
    """
    log_files = []
    
    # Main log file
    main_log = os.path.join(model_dir, "log.txt")
    if os.path.exists(main_log):
        log_files.append(main_log)
    
    # Resume log files - pattern: log_resume_epoch_{n_epoch}_iter_0
    resume_pattern = os.path.join(model_dir, "log_resume_epoch_*_iter_0.txt")
    resume_logs = glob.glob(resume_pattern)
    
    # Sort resume logs by epoch number
    resume_logs_with_epoch = []
    for log_path in resume_logs:
        filename = os.path.basename(log_path)
        epoch_match = re.search(r'log_resume_epoch_(\d+)_iter_0', filename)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            resume_logs_with_epoch.append((epoch_num, log_path))
    
    # Sort by epoch number and add to log_files
    resume_logs_with_epoch.sort(key=lambda x: x[0])
    for epoch_num, log_path in resume_logs_with_epoch:
        log_files.append(log_path)
    
    return log_files

def parse_training_log(model_dir):
    """
    Parse all training log files (including resume logs) and extract loss information
    
    Returns:
    - epochs: list of epoch numbers
    - train_losses: list of training losses per epoch
    - valid_losses: list of validation losses per epoch
    - best_valid_losses: list of best validation losses per epoch
    """
    
    epochs = []
    train_losses = []
    valid_losses = []
    best_valid_losses = []
    
    # Find all log files
    log_files = find_log_files(model_dir)
    
    if not log_files:
        print(f"Warning: No log files found in {model_dir}")
        return epochs, train_losses, valid_losses, best_valid_losses
    
    print(f"Found {len(log_files)} log file(s):")
    for log_file in log_files:
        print(f"  - {os.path.basename(log_file)}")
    
    # Patterns to match the log lines
    train_pattern = r'train \[(\d+)/\d+\] Loss: ([\d.]+), Best valid: ([\d.]+|inf)'
    valid_pattern = r'valid \[(\d+)/\d+\] Loss: ([\d.]+), Best valid: ([\d.]+|inf)'
    
    # Parse each log file
    for log_path in log_files:
        print(f"Parsing: {os.path.basename(log_path)}")
        
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Match training epoch summary
                train_match = re.search(train_pattern, line)
                if train_match:
                    epoch = int(train_match.group(1))
                    loss = float(train_match.group(2))
                    best_valid = float(train_match.group(3)) if train_match.group(3) != 'inf' else np.inf
                    
                    epochs.append(epoch)
                    train_losses.append(loss)
                    best_valid_losses.append(best_valid)
                
                # Match validation epoch summary
                valid_match = re.search(valid_pattern, line)
                if valid_match:
                    epoch = int(valid_match.group(1))
                    loss = float(valid_match.group(2))
                    
                    valid_losses.append(loss)
    
    # Sort by epoch number (in case files were parsed out of order)
    if epochs:
        combined = list(zip(epochs, train_losses, valid_losses, best_valid_losses))
        combined.sort(key=lambda x: x[0])
        epochs, train_losses, valid_losses, best_valid_losses = zip(*combined)
        epochs = list(epochs)
        train_losses = list(train_losses)
        valid_losses = list(valid_losses)
        best_valid_losses = list(best_valid_losses)
    
    return epochs, train_losses, valid_losses, best_valid_losses

def plot_training_curves(epochs, train_losses, valid_losses, best_valid_losses, 
                        model_name, save_path=None):
    """
    Plot training and validation loss curves
    """
    
    plt.figure(figsize=(9, 6))
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.plot(epochs, best_valid_losses, 'g--', label='Best Validation Loss', linewidth=1, alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (RMSE)')
    plt.title(f'Training Progress - Model {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
        
    # Find best epoch
    best_epoch = epochs[np.argmin(valid_losses)]
    best_loss = min(valid_losses)
    
    # Add annotation for best model
    plt.annotate(f'Best: Epoch {best_epoch}\nLoss: {best_loss:.6f}', 
                xy=(best_epoch, best_loss), 
                xytext=(best_epoch + len(epochs)*0.1, best_loss + (max(valid_losses) - min(valid_losses))*0.1),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curve saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY - Model {model_name}")
    print(f"{'='*60}")
    print(f"Total epochs: {len(epochs)}")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {valid_losses[-1]:.6f}")
    print(f"Best validation loss: {min(valid_losses):.6f} (epoch {best_epoch})")
    print(f"Initial training loss: {train_losses[0]:.6f}")
    print(f"Initial validation loss: {valid_losses[0]:.6f}")
    print(f"Loss reduction: {train_losses[0] - train_losses[-1]:.6f} (training)")
    print(f"Loss reduction: {valid_losses[0] - valid_losses[-1]:.6f} (validation)")
    
    # Check for overfitting
    if valid_losses[-1] > min(valid_losses) * 1.1:
        print(f"⚠️  Potential overfitting detected!")
        print(f"   Current validation loss is {(valid_losses[-1] / min(valid_losses) - 1) * 100:.1f}% higher than best")
    else:
        print(f"✅ Training appears stable")

def plot_loss_distribution(epochs, train_losses, valid_losses, model_name, save_path=None):
    """
    Plot histogram of loss values to understand distribution
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training loss distribution
    ax1.hist(train_losses, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(train_losses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(train_losses):.6f}')
    ax1.axvline(np.median(train_losses), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(train_losses):.6f}')
    ax1.set_xlabel('Training Loss')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Training Loss Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation loss distribution
    ax2.hist(valid_losses, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax2.axvline(np.mean(valid_losses), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_losses):.6f}')
    ax2.axvline(np.median(valid_losses), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(valid_losses):.6f}')
    ax2.set_xlabel('Validation Loss')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Validation Loss Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Loss Distributions - Model {model_name}')
    plt.tight_layout()
    
    if save_path:
        dist_save_path = save_path.replace('.png', '_distribution.png')
        plt.savefig(dist_save_path, dpi=300, bbox_inches='tight')
        print(f"Loss distribution saved to: {dist_save_path}")
    
    plt.show()

def plot_multi_model_comparison(model_names, save_path=None):
    """
    Plot best validation loss curves for multiple models
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    all_model_data = []
    
    for i, model_name in enumerate(model_names):
        model_dir = f"data/gnn_dyn_model/{model_name}"
        
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory '{model_dir}' does not exist, skipping...")
            continue
            
        print(f"\nProcessing model {model_name}...")
        epochs, train_losses, valid_losses, best_valid_losses = parse_training_log(model_dir)
        
        if not epochs:
            print(f"Warning: No training data found for {model_name}, skipping...")
            continue
        
        color = colors[i % len(colors)]
        
        # Plot best validation loss curve
        plt.plot(epochs, best_valid_losses, 
                color=color, linewidth=2, 
                label=f'{model_name}\n(Best: {min(best_valid_losses):.6f})')
        
        # Store data for summary
        all_model_data.append({
            'name': model_name,
            'epochs': epochs,
            'best_valid_losses': best_valid_losses,
            'final_best': best_valid_losses[-1],
            'overall_best': min(best_valid_losses),
            'total_epochs': len(epochs)
        })
    
    plt.xlabel('Epoch')
    plt.ylabel('Best Validation Loss (RMSE)')
    plt.title('Multi-Model Comparison: Best Validation Loss Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nMulti-model comparison saved to: {save_path}")
    
    plt.show()
    
    # Print comparison summary
    if all_model_data:
        print(f"\n{'='*80}")
        print("MULTI-MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Sort by overall best validation loss
        all_model_data.sort(key=lambda x: x['overall_best'])
        
        print(f"{'Rank':<4} {'Model Name':<25} {'Total Epochs':<12} {'Overall Best':<15} {'Final Best':<15}")
        print("-" * 80)
        
        for rank, data in enumerate(all_model_data, 1):
            print(f"{rank:<4} {data['name']:<25} {data['total_epochs']:<12} "
                  f"{data['overall_best']:<15.6f} {data['final_best']:<15.6f}")
        
        best_model = all_model_data[0]
        print(f"\n🏆 Best performing model: {best_model['name']}")
        print(f"   Best validation loss: {best_model['overall_best']:.6f}")
        print(f"   Total epochs trained: {best_model['total_epochs']}")

def main():
    parser = argparse.ArgumentParser(description='Visualize GNN training loss curves')
    parser.add_argument('--name', type=str,
                       help='Single model name (e.g., 2025-06-01-15-33-02-230868 or custom_model_name)')
    parser.add_argument('--compare', type=str, nargs='+',
                       help='Multiple model names to compare (e.g., --compare model1 model2 model3)')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to files instead of just displaying')
    args = parser.parse_args()
    
    if args.compare:
        # Multi-model comparison mode
        model_names = args.compare
        print(f"Comparing {len(model_names)} models...")
        
        save_path = None
        if args.save_plots:
            save_path = f"data/plots/multi_model_comparison.png"
        
        plot_multi_model_comparison(model_names, save_path)
        
    elif args.name:
        # Single model mode
        model_name = args.name
        model_dir = f"data/gnn_dyn_model/{model_name}"
        
        if not os.path.exists(model_dir):
            print(f"Error: Model directory '{model_dir}' does not exist!")
            print(f"Please check the model name and ensure the model directory exists.")
            return
        
        print(f"Analyzing model: {model_name}")
        
        # Parse the log files
        epochs, train_losses, valid_losses, best_valid_losses = parse_training_log(model_dir)
        
        if not epochs:
            print("Error: No training data found in log files.")
            return
        
        print(f"Found training data for {len(epochs)} epochs")
        
        # Set up save paths if requested
        save_path = None
        if args.save_plots:
            save_path = f"data/plots/{model_name}_training_curves.png"
        
        # Create visualizations
        plot_training_curves(epochs, train_losses, valid_losses, best_valid_losses, 
                            model_name, save_path)
        
        plot_loss_distribution(epochs, train_losses, valid_losses, model_name, save_path)
        
        # Additional analysis
        print(f"\n{'='*60}")
        print("TRAINING ANALYSIS")
        print(f"{'='*60}")
        
        # Convergence analysis
        last_10_epochs = train_losses[-10:] if len(train_losses) >= 10 else train_losses
        train_std = np.std(last_10_epochs)
        
        last_10_valid = valid_losses[-10:] if len(valid_losses) >= 10 else valid_losses
        valid_std = np.std(last_10_valid)
        
        print(f"Training stability (last 10 epochs std): {train_std:.6f}")
        print(f"Validation stability (last 10 epochs std): {valid_std:.6f}")
        
        if train_std < 0.001 and valid_std < 0.001:
            print("✅ Training has converged (low variation in recent epochs)")
        else:
            print("⚠️  Training may still be improving (high variation in recent epochs)")
        
        # Learning rate analysis
        print(f"\nLearning rate appears to be: 0.001 (constant)")
        
        # Recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        
        final_train = train_losses[-1]
        final_valid = valid_losses[-1]
        
        if final_valid < final_train:
            print("✅ Good generalization - validation loss is lower than training loss")
        elif final_valid > final_train * 1.5:
            print("⚠️  Possible overfitting - consider regularization or early stopping")
        else:
            print("✅ Reasonable generalization")
            
        if valid_std > 0.001:
            print("💡 Consider training for more epochs to reach convergence")
        elif min(valid_losses) == valid_losses[-1]:
            print("💡 Training stopped at optimal point")
        else:
            print("💡 Consider early stopping or model from best validation epoch")
    
    else:
        print("Error: Please specify either --name for single model analysis or --compare for multi-model comparison")
        parser.print_help()

if __name__ == "__main__":
    main() 