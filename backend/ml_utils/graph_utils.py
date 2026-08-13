import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import numpy as np
import scipy.stats as stats

THEME_COLORS = {
    'background': '#0a0a1a',  # rich-black
    'surface': '#1a1a2a',    # raisin-black
    'surface_alt': '#282a3a', # raisin-black-alt
    'text': '#f0f0f0',       # anti-flash-white
    'primary': '#ff8c00',    # dark-orange
    'secondary': '#daa520',  # goldenrod
    'accent1': '#ffa333',    # dark-orange-600
    'accent2': '#e4b84a',    # goldenrod-600
    'grid': '#3b3b60',       # raisin-black-600
    'line': '#f3f3f3'        # anti-flash-white-600
}

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': THEME_COLORS['background'],
    'axes.facecolor': THEME_COLORS['surface'],
    'axes.edgecolor': THEME_COLORS['grid'],
    'axes.labelcolor': THEME_COLORS['text'],
    'axes.titlecolor': THEME_COLORS['text'],
    'xtick.color': THEME_COLORS['text'],
    'ytick.color': THEME_COLORS['text'],
    'text.color': THEME_COLORS['text'],
    'grid.color': THEME_COLORS['grid'],
    'grid.alpha': 0.3,
    'axes.grid': True,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10
})

def save_confusion_matrix_graph(y_true, y_pred, path):
    plt.figure(figsize=(8, 6), facecolor=THEME_COLORS['background'])
    ax = plt.gca()
    ax.set_facecolor(THEME_COLORS['surface'])
    
    cm = confusion_matrix(y_true, y_pred)
    
    from matplotlib.colors import LinearSegmentedColormap
    colors = [THEME_COLORS['surface'], THEME_COLORS['primary']]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                annot_kws={'color': THEME_COLORS['text'], 'fontweight': 'bold'},
                cbar_kws={'label': 'Count'})
    
    plt.title("Confusion Matrix", color=THEME_COLORS['text'], fontweight='bold', pad=20)
    plt.xlabel("Predicted", color=THEME_COLORS['text'], fontweight='bold')
    plt.ylabel("Actual", color=THEME_COLORS['text'], fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, facecolor=THEME_COLORS['background'], edgecolor='none', dpi=300)
    plt.close()

def save_roc_curve_graph(y_true, y_proba, path):
    plt.figure(figsize=(8, 6), facecolor=THEME_COLORS['background'])
    ax = plt.gca()
    ax.set_facecolor(THEME_COLORS['surface'])
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color=THEME_COLORS['primary'], linewidth=3, 
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color=THEME_COLORS['grid'], linestyle='--', linewidth=2, alpha=0.8)
    
    plt.title("ROC Curve", color=THEME_COLORS['text'], fontweight='bold', pad=20)
    plt.xlabel("False Positive Rate", color=THEME_COLORS['text'], fontweight='bold')
    plt.ylabel("True Positive Rate", color=THEME_COLORS['text'], fontweight='bold')
    plt.legend(facecolor=THEME_COLORS['surface_alt'], edgecolor=THEME_COLORS['grid'])
    plt.grid(True, alpha=0.3, color=THEME_COLORS['grid'])
    plt.tight_layout()
    plt.savefig(path, facecolor=THEME_COLORS['background'], edgecolor='none', dpi=300)
    plt.close()

def save_precision_recall_graph(y_true, y_proba, path):
    plt.figure(figsize=(8, 6), facecolor=THEME_COLORS['background'])
    ax = plt.gca()
    ax.set_facecolor(THEME_COLORS['surface'])
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, marker='o', markersize=4, linewidth=3,
             color=THEME_COLORS['secondary'], markerfacecolor=THEME_COLORS['accent2'],
             label=f'PR Curve (AUC = {pr_auc:.3f})')
    
    plt.title("Precision-Recall Curve", color=THEME_COLORS['text'], fontweight='bold', pad=20)
    plt.xlabel("Recall", color=THEME_COLORS['text'], fontweight='bold')
    plt.ylabel("Precision", color=THEME_COLORS['text'], fontweight='bold')
    plt.legend(facecolor=THEME_COLORS['surface_alt'], edgecolor=THEME_COLORS['grid'])
    plt.grid(True, alpha=0.3, color=THEME_COLORS['grid'])
    plt.tight_layout()
    plt.savefig(path, facecolor=THEME_COLORS['background'], edgecolor='none', dpi=300)
    plt.close()

def save_multiclass_roc_curve_graph(y_true, y_proba, path):
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    y_true_bin = label_binarize(y_true, classes=classes)
    if n_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
    
    plt.figure(figsize=(10, 8), facecolor=THEME_COLORS['background'])
    ax = plt.gca()
    ax.set_facecolor(THEME_COLORS['surface'])
    
    class_colors = [THEME_COLORS['primary'], THEME_COLORS['secondary'], 
                   THEME_COLORS['accent1'], THEME_COLORS['accent2']]
    
    if n_classes > len(class_colors):
        additional_colors = plt.cm.Set1(np.linspace(0, 1, n_classes - len(class_colors)))
        class_colors.extend([f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' 
                            for c in additional_colors])
    
    for i in range(n_classes):
        if n_classes == 2 and i == 0:
            continue
        
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=class_colors[i % len(class_colors)], lw=3,
                label=f'Class {classes[i]} (AUC = {roc_auc:.3f})')
    
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    plt.plot(fpr_micro, tpr_micro,
             label=f'Micro-average (AUC = {roc_auc_micro:.3f})',
             color=THEME_COLORS['line'], linestyle=':', linewidth=3)
    
    plt.plot([0, 1], [0, 1], color=THEME_COLORS['grid'], linestyle='--', lw=2, alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', color=THEME_COLORS['text'], fontweight='bold')
    plt.ylabel('True Positive Rate', color=THEME_COLORS['text'], fontweight='bold')
    plt.title('Multiclass ROC Curves', color=THEME_COLORS['text'], fontweight='bold', pad=20)
    plt.legend(loc="lower right", facecolor=THEME_COLORS['surface_alt'], edgecolor=THEME_COLORS['grid'])
    plt.grid(True, alpha=0.3, color=THEME_COLORS['grid'])
    plt.tight_layout()
    plt.savefig(path, facecolor=THEME_COLORS['background'], edgecolor='none', dpi=300)
    plt.close()

def save_multiclass_precision_recall_graph(y_true, y_proba, path):
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    y_true_bin = label_binarize(y_true, classes=classes)
    if n_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
    
    plt.figure(figsize=(10, 8), facecolor=THEME_COLORS['background'])
    ax = plt.gca()
    ax.set_facecolor(THEME_COLORS['surface'])
    
    class_colors = [THEME_COLORS['primary'], THEME_COLORS['secondary'], 
                   THEME_COLORS['accent1'], THEME_COLORS['accent2']]
    
    if n_classes > len(class_colors):
        additional_colors = plt.cm.Set1(np.linspace(0, 1, n_classes - len(class_colors)))
        class_colors.extend([f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' 
                            for c in additional_colors])
    
    for i in range(n_classes):
        if n_classes == 2 and i == 0:
            continue
            
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, color=class_colors[i % len(class_colors)], lw=3,
                label=f'Class {classes[i]} (AUC = {pr_auc:.3f})')
    
    precision_micro, recall_micro, _ = precision_recall_curve(
        y_true_bin.ravel(), y_proba.ravel())
    pr_auc_micro = auc(recall_micro, precision_micro)
    
    plt.plot(recall_micro, precision_micro,
             label=f'Micro-average (AUC = {pr_auc_micro:.3f})',
             color=THEME_COLORS['line'], linestyle=':', linewidth=3)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', color=THEME_COLORS['text'], fontweight='bold')
    plt.ylabel('Precision', color=THEME_COLORS['text'], fontweight='bold')
    plt.title('Multiclass Precision-Recall Curves', color=THEME_COLORS['text'], fontweight='bold', pad=20)
    plt.legend(facecolor=THEME_COLORS['surface_alt'], edgecolor=THEME_COLORS['grid'])
    plt.grid(True, alpha=0.3, color=THEME_COLORS['grid'])
    plt.tight_layout()
    plt.savefig(path, facecolor=THEME_COLORS['background'], edgecolor='none', dpi=300)
    plt.close()

def save_residual_plot(y_true, y_pred, path):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6), facecolor=THEME_COLORS['background'])
    ax = plt.gca()
    ax.set_facecolor(THEME_COLORS['surface'])
    
    plt.scatter(y_pred, residuals, alpha=0.7, color=THEME_COLORS['primary'], 
               s=50, edgecolors=THEME_COLORS['accent1'], linewidth=0.5)
    plt.axhline(0, color=THEME_COLORS['secondary'], linestyle='--', linewidth=2)
    
    plt.title("Residual Plot", color=THEME_COLORS['text'], fontweight='bold', pad=20)
    plt.xlabel("Predicted Values", color=THEME_COLORS['text'], fontweight='bold')
    plt.ylabel("Residuals", color=THEME_COLORS['text'], fontweight='bold')
    plt.grid(True, alpha=0.3, color=THEME_COLORS['grid'])
    plt.tight_layout()
    plt.savefig(path, facecolor=THEME_COLORS['background'], edgecolor='none', dpi=300)
    plt.close()

def save_actual_vs_predicted_plot(y_true, y_pred, path):
    plt.figure(figsize=(8, 6), facecolor=THEME_COLORS['background'])
    ax = plt.gca()
    ax.set_facecolor(THEME_COLORS['surface'])
    
    plt.scatter(y_true, y_pred, alpha=0.7, color=THEME_COLORS['primary'], 
               s=50, edgecolors=THEME_COLORS['accent1'], linewidth=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             color=THEME_COLORS['secondary'], linestyle='--', linewidth=2)
    
    plt.title("Actual vs Predicted", color=THEME_COLORS['text'], fontweight='bold', pad=20)
    plt.xlabel("Actual Values", color=THEME_COLORS['text'], fontweight='bold')
    plt.ylabel("Predicted Values", color=THEME_COLORS['text'], fontweight='bold')
    plt.grid(True, alpha=0.3, color=THEME_COLORS['grid'])
    plt.tight_layout()
    plt.savefig(path, facecolor=THEME_COLORS['background'], edgecolor='none', dpi=300)
    plt.close()

def save_error_distribution_plot(y_true, y_pred, path):
    errors = y_true - y_pred
    plt.figure(figsize=(8, 6), facecolor=THEME_COLORS['background'])
    ax = plt.gca()
    ax.set_facecolor(THEME_COLORS['surface'])
    
    n, bins, patches = plt.hist(errors, bins=30, alpha=0.7, color=THEME_COLORS['primary'],
                               edgecolor=THEME_COLORS['accent1'], linewidth=1)
    
    from scipy import stats
    kde = stats.gaussian_kde(errors)
    x_range = np.linspace(errors.min(), errors.max(), 100)
    kde_values = kde(x_range)
    plt.plot(x_range, kde_values * len(errors) * (bins[1] - bins[0]), 
             color=THEME_COLORS['secondary'], linewidth=3, label='KDE')
    
    plt.title("Prediction Error Distribution", color=THEME_COLORS['text'], fontweight='bold', pad=20)
    plt.xlabel("Error", color=THEME_COLORS['text'], fontweight='bold')
    plt.ylabel("Frequency", color=THEME_COLORS['text'], fontweight='bold')
    plt.legend(facecolor=THEME_COLORS['surface_alt'], edgecolor=THEME_COLORS['grid'])
    plt.grid(True, alpha=0.3, color=THEME_COLORS['grid'])
    plt.tight_layout()
    plt.savefig(path, facecolor=THEME_COLORS['background'], edgecolor='none', dpi=300)
    plt.close()

def save_qq_plot(y_true, y_pred, path):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6), facecolor=THEME_COLORS['background'])
    ax = plt.gca()
    ax.set_facecolor(THEME_COLORS['surface'])
    
    stats.probplot(residuals, dist="norm", plot=plt)
    
    line = ax.get_lines()
    if len(line) >= 2:
        line[0].set_markerfacecolor(THEME_COLORS['primary'])
        line[0].set_markeredgecolor(THEME_COLORS['accent1'])
        line[0].set_markersize(6)
        line[1].set_color(THEME_COLORS['secondary'])
        line[1].set_linewidth(2)
    
    plt.title("Q-Q Plot of Residuals", color=THEME_COLORS['text'], fontweight='bold', pad=20)
    plt.xlabel("Theoretical Quantiles", color=THEME_COLORS['text'], fontweight='bold')
    plt.ylabel("Sample Quantiles", color=THEME_COLORS['text'], fontweight='bold')
    plt.grid(True, alpha=0.3, color=THEME_COLORS['grid'])
    plt.tight_layout()
    plt.savefig(path, facecolor=THEME_COLORS['background'], edgecolor='none', dpi=300)
    plt.close()
    
def save_feature_importance_graph(importances, feature_names, save_path):
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def reset_matplotlib_theme():
    plt.rcdefaults()
    plt.style.use('default')