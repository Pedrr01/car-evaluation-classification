import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report,
                             precision_recall_fscore_support)
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')


ALGO_NAME = "K-Nearest Neighbors (KNN)"
CLASSIFIER_CLASS = KNeighborsClassifier
PARAM_COMBOS = [
    {"n_neighbors": 3, "weights": "uniform", "metric": "minkowski", "p": 2},
    {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski", "p": 2},
    {"n_neighbors": 7, "weights": "distance", "metric": "minkowski", "p": 2},
]

feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
target_name = 'class'
categories = {
    'buying': ['low', 'med', 'high', 'vhigh'],
    'maint': ['low', 'med', 'high', 'vhigh'],
    'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high']
}


def get_param_label(params):
    if 'n_neighbors' in params:
        return f"k={params['n_neighbors']}, {params['weights']}"
    elif 'var_smoothing' in params:
        return f"var_smoothing={params['var_smoothing']:.0e}"
    return str(params)


def load_and_preprocess_data(file_path=None):
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, '..', 'car.data'),
            os.path.join(script_dir, 'car.data'),
            'car.data',
            '/home/workdir/car.data'
        ]
        for cand in candidates:
            if os.path.exists(cand):
                file_path = cand
                break
        else:
            file_path = 'car.data'  

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, header=None, names=feature_names + [target_name])
        data_source = "real (car.data)"
        oe = OrdinalEncoder(categories=[categories[col] for col in feature_names])
        X = oe.fit_transform(df[feature_names])
        le = LabelEncoder()
        y = le.fit_transform(df[target_name])
        classes_ = le.classes_
    else:
        data_source = "sintético (baseado em Car Evaluation)"
        np.random.seed(42)
        n = 200
        X_list = []
        for col, cats in categories.items():
            n_cats = len(cats)
            X_list.append(np.random.randint(0, n_cats, n))
        X = np.column_stack(X_list).astype(float)
        
        safety = X[:, 5].astype(int)
        buying = X[:, 0].astype(int)
        persons = X[:, 3].astype(int)
        score = (safety * 2 + (3 - buying) + (persons > 0).astype(int))
        y = np.clip((score / 3).astype(int), 0, 3)
        y = np.where(np.random.rand(n) < 0.12, np.random.randint(0, 4, n), y)  
        le = LabelEncoder()
        le.classes_ = np.asarray(['unacc', 'acc', 'good', 'vgood'])
        classes_ = le.classes_
        oe = None

    print("=" * 65)
    print(f"DATASET: Car Evaluation")
    print(f"Número de exemplos: {len(y)}")
    print(f"Número de atributos: {X.shape[1]}")
    print(f"Número de classes: {len(classes_)}")
    print(f"Classes: {list(classes_)}")
    majority_idx = np.bincount(y.astype(int)).argmax()
    majority_class = classes_[majority_idx]
    majority_count = np.bincount(y.astype(int))[majority_idx]
    print(f"Classe majoritária: '{majority_class}' com {majority_count} exemplos "
          f"({majority_count / len(y) * 100:.1f}%)")
    print("=" * 65 + "\n")
    return X, y, le, data_source


def evaluate_with_cv(clf, X, y, param_combos, cv=10, random_state=42):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []

    for i, params in enumerate(param_combos):
        print(f"\nCombinação de Parâmetros {i+1}/{len(param_combos)}: {params}")
        clf.set_params(**params)

        fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

            fold_metrics['accuracy'].append(acc)
            fold_metrics['precision'].append(prec)
            fold_metrics['recall'].append(rec)
            fold_metrics['f1'].append(f1)

        means = {m: np.mean(v) for m, v in fold_metrics.items()}
        stds = {m: np.std(v, ddof=1) for m, v in fold_metrics.items()}

        print(f"   Acurácia:     {means['accuracy']:.4f} ± {stds['accuracy']:.4f}")
        print(f"   Precisão (w): {means['precision']:.4f} ± {stds['precision']:.4f}")
        print(f"   Recall (w):   {means['recall']:.4f} ± {stds['recall']:.4f}")
        print(f"   F1-score (w): {means['f1']:.4f} ± {stds['f1']:.4f}")

        results.append({
            'params': params,
            'means': means,
            'stds': stds,
            'fold_metrics': fold_metrics
        })

    return results


def plot_results(results, cm, label_encoder, y_test, y_pred_test, algo_name, data_source):
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=0.95)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    ax0 = fig.add_subplot(gs[0, :])
    combo_labels = [get_param_label(r['params']) for r in results]
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Acurácia', 'Precisão (w)', 'Recall (w)', 'F1-score (w)']
    x = np.arange(len(combo_labels))
    width = 0.18
    colors = sns.color_palette("husl", 4)

    for j, (met, lab) in enumerate(zip(metrics_names, metric_labels)):
        means = [r['means'][met] for r in results]
        stds = [r['stds'][met] for r in results]
        bars = ax0.bar(x + j * width, means, width, label=lab, color=colors[j],
                       yerr=stds, capsize=4, error_kw={'linewidth': 1.2, 'alpha': 0.7})

    ax0.set_ylabel('Score (média ± desvio padrão)', fontsize=11)
    ax0.set_xlabel('Combinações de Hiperparâmetros', fontsize=11)
    ax0.set_title(f'{algo_name} — Validação Cruzada 10-Fold Estratificada\n{data_source}', fontsize=13, fontweight='bold', pad=10)
    ax0.set_xticks(x + width * 1.5)
    ax0.set_xticklabels(combo_labels, rotation=20, ha='right', fontsize=9)
    ax0.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax0.set_ylim(0, 1.08)
    ax0.grid(axis='y', alpha=0.3, linestyle='--')
    ax0.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)  

    ax1 = fig.add_subplot(gs[1, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
                annot_kws={'size': 14, 'weight': 'bold'}, cbar_kws={'shrink': 0.8})
    ax1.set_xlabel('Classe Predita', fontsize=11)
    ax1.set_ylabel('Classe Verdadeira', fontsize=11)
    ax1.set_title('Matriz de Confusão — Conjunto de Teste (20%)', fontsize=12, fontweight='bold')

    ax2 = fig.add_subplot(gs[1, 1])
    prec_per, rec_per, f1_per, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average=None, zero_division=0
    )
    bars = ax2.bar(label_encoder.classes_, f1_per, color=sns.color_palette("husl", 4), edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('F1-score', fontsize=11)
    ax2.set_xlabel('Classe', fontsize=11)
    ax2.set_title('F1-score por Classe no Conjunto de Teste', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.08)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, f1_per):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.suptitle(f'Resultados Padronizados — {algo_name} no Dataset Car Evaluation', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    graficos_dir = "graficos"
    os.makedirs(graficos_dir, exist_ok=True)

    safe_name = algo_name.replace(" (KNN)", "").replace(" (GaussianNB)", "").replace(" ", "_").replace("-", "_")
    filename = os.path.join(graficos_dir, f"{safe_name}.png")
    plt.savefig(filename, dpi=160, bbox_inches='tight', facecolor='white')
    plt.show()


def main():
    print("\n" + "=" * 65)
    print(f"PROJETO 2ª VA — INTELIGÊNCIA ARTIFICIAL")
    print(f"Algoritmo: {ALGO_NAME}")
    print(f"Procedimento: 10-fold CV Estratificada + Grid-like (3+ combinações)")
    print("=" * 65 + "\n")

    X, y, label_encoder, data_source = load_and_preprocess_data('car.data')

    print("Validação Cruzada 10-Fold Estratificada:\n")
    clf = CLASSIFIER_CLASS()
    results = evaluate_with_cv(clf, X, y, PARAM_COMBOS, cv=10, random_state=42)

    print("\n" + "=" * 65)
    print("RESUMO DOS RESULTADOS — VALIDAÇÃO CRUZADA 10-FOLD")
    print("=" * 65)
    summary_data = {
        'Combinação': [get_param_label(r['params']) for r in results],
        'Acurácia': [f"{r['means']['accuracy']:.4f} ± {r['stds']['accuracy']:.4f}" for r in results],
        'Precisão (w)': [f"{r['means']['precision']:.4f} ± {r['stds']['precision']:.4f}" for r in results],
        'Recall (w)': [f"{r['means']['recall']:.4f} ± {r['stds']['recall']:.4f}" for r in results],
        'F1-score (w)': [f"{r['means']['f1']:.4f} ± {r['stds']['f1']:.4f}" for r in results],
    }
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    print("=" * 65)

    best_idx = int(np.argmax([r['means']['f1'] for r in results]))
    best_result = results[best_idx]
    print(f"\nMELHOR COMBINAÇÃO: {get_param_label(best_result['params'])}")
    print(f"   F1-score médio: {best_result['means']['f1']:.4f} ± {best_result['stds']['f1']:.4f}")

    print("\nDivisão hold-out (80/20 estratificada) para avaliação final:")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    best_clf = CLASSIFIER_CLASS(**best_result['params'])
    best_clf.fit(X_train, y_train)
    y_pred_test = best_clf.predict(X_test)

    print("\n" + "=" * 65)
    print("AVALIAÇÃO FINAL NO CONJUNTO DE TESTE (20%) — MELHOR MODELO")
    print("=" * 65)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_rec = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    print(f"Acurácia:     {test_acc:.4f}")
    print(f"Precisão (w): {test_prec:.4f}")
    print(f"Recall (w):   {test_rec:.4f}")
    print(f"F1-score (w): {test_f1:.4f}\n")

    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred_test,
                                target_names=label_encoder.classes_,
                                zero_division=0, digits=4))

    cm = confusion_matrix(y_test, y_pred_test)
    print("Matriz de Confusão (valores absolutos):")
    print(cm)
    print("=" * 65)

    plot_results(results, cm, label_encoder, y_test, y_pred_test, ALGO_NAME, data_source)


if __name__ == "__main__":
    main()
