import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

def parse_training_log(log_path):
    """
    Parse training log file and extract loss information
    
    Returns:
    - epochs: list of epoch numbers
    - train_losses: list of training losses per epoch
    - valid_losses: list of validation losses per epoch
    - train_rmse: list of training RMSE per epoch (since logs show sqrt of MSE)
    - valid_rmse: list of validation RMSE per epoch
    """
    
    epochs = []
    train_losses = []
    valid_losses = []
    best_valid_losses = []
    
    # Patterns to match the log lines
    train_pattern = r'train \[(\d+)/\d+\] Loss: ([\d.]+), Best valid: ([\d.]+|inf)'
    valid_pattern = r'valid \[(\d+)/\d+\] Loss: ([\d.]+), Best valid: ([\d.]+|inf)'
    
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
    
    return epochs, train_losses, valid_losses, best_valid_losses

def plot_training_curves(epochs, train_losses, valid_losses, best_valid_losses, 
                        model_timestamp, save_path=None):
    """
    Plot training and validation loss curves
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Training and Validation Loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.plot(epochs, best_valid_losses, 'g--', label='Best Validation Loss', linewidth=1, alpha=0.7)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (RMSE)')
    ax1.set_title(f'Training Progress - Model {model_timestamp}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: Validation Loss (zoomed)
    ax2.plot(epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2)
    ax2.plot(epochs, best_valid_losses, 'g--', label='Best Validation Loss', linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss (RMSE)')
    ax2.set_title(f'Validation Loss Detail - Model {model_timestamp}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Find best epoch
    best_epoch = epochs[np.argmin(valid_losses)]
    best_loss = min(valid_losses)
    
    # Add annotation for best model
    ax2.annotate(f'Best: Epoch {best_epoch}\nLoss: {best_loss:.6f}', 
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
    print(f"TRAINING SUMMARY - Model {model_timestamp}")
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

def plot_loss_distribution(epochs, train_losses, valid_losses, model_timestamp, save_path=None):
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
    
    plt.suptitle(f'Loss Distributions - Model {model_timestamp}')
    plt.tight_layout()
    
    if save_path:
        dist_save_path = save_path.replace('.png', '_distribution.png')
        plt.savefig(dist_save_path, dpi=300, bbox_inches='tight')
        print(f"Loss distribution saved to: {dist_save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize GNN training loss curves')
    parser.add_argument('--timestamp', type=str, required=True,
                       help='Model timestamp (e.g., 2025-06-01-15-33-02-230868)')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to files instead of just displaying')
    args = parser.parse_args()
    
    timestamp = args.timestamp
    log_path = f"data/gnn_dyn_model/{timestamp}/log.txt"
    
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        print(f"Please check the timestamp and ensure the training log exists.")
        return
    
    print(f"Parsing training log: {log_path}")
    
    # Parse the log file
    epochs, train_losses, valid_losses, best_valid_losses = parse_training_log(log_path)
    
    if not epochs:
        print("Error: No training data found in log file.")
        return
    
    print(f"Found training data for {len(epochs)} epochs")
    
    # Set up save paths if requested
    save_path = None
    if args.save_plots:
        save_path = f"data/plots/{timestamp}_training_curves.png"
    
    # Create visualizations
    plot_training_curves(epochs, train_losses, valid_losses, best_valid_losses, 
                        timestamp, save_path)
    
    plot_loss_distribution(epochs, train_losses, valid_losses, timestamp, save_path)
    
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

if __name__ == "__main__":
    main() 