"""
plot_main_results.py — 主实验结果可视化

对比三个模型在 Otto Group 数据集上的表现：
  - Accuracy
  - AUC-ROC
  - Training Time
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def load_results():
    """读取三个模型的主实验结果"""
    results = {}
    for model in ['xgboost', 'random_forest', 'xrfm']:
        with open(f'results/run_{model}.json') as f:
            results[model] = json.load(f)
    return results


def plot_comparison(results):
    """画对比图"""
    models = ['XGBoost', 'Random Forest', 'xRFM']
    keys = ['xgboost', 'random_forest', 'xrfm']
    
    accuracy = [results[k]['accuracy'] for k in keys]
    auc_roc = [results[k]['auc_roc'] for k in keys]
    train_time = [results[k]['train_time'] for k in keys]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Accuracy
    axes[0].bar(models, accuracy, color=colors)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Test Accuracy')
    axes[0].set_ylim(0.75, 0.80)
    for i, v in enumerate(accuracy):
        axes[0].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    
    # AUC-ROC
    axes[1].bar(models, auc_roc, color=colors)
    axes[1].set_ylabel('AUC-ROC')
    axes[1].set_title('AUC-ROC Score')
    axes[1].set_ylim(0.95, 0.97)
    for i, v in enumerate(auc_roc):
        axes[1].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    
    # Training Time
    axes[2].bar(models, train_time, color=colors)
    axes[2].set_ylabel('Time (seconds)')
    axes[2].set_title('Training Time')
    for i, v in enumerate(train_time):
        axes[2].text(i, v + 0.5, f'{v:.2f}s', ha='center', va='bottom')
    
    plt.suptitle('Otto Group Dataset — Main Experiment Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/main_experiment_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: results/main_experiment_comparison.png")
    plt.close()


if __name__ == "__main__":
    results = load_results()
    plot_comparison(results)
