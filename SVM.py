import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import os

from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_FOLDS = 10

COL_NAMES = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
LOCAL_FILE = "car.data" 
GRAFICO_SAIDA = "svm_graficos_resultados.png"  


def carregar_dataset():
    """Carrega o dataset do UCI; se nao houver internet, usa arquivo local."""
    if os.path.exists(LOCAL_FILE):
        df = pd.read_csv(LOCAL_FILE, names=COL_NAMES)
        print(f"[INFO] Dataset carregado do arquivo local '{LOCAL_FILE}'.")
    else:
        try:
            urllib.request.urlretrieve(UCI_URL, LOCAL_FILE)
            df = pd.read_csv(LOCAL_FILE, names=COL_NAMES)
            print("[INFO] Dataset baixado diretamente do repositorio UCI.")
        except Exception as e:
            raise RuntimeError(
                "Nao foi possivel obter o dataset. Baixe manualmente o arquivo "
                "'car.data' em https://archive.ics.uci.edu/ml/machine-learning-databases/car/ "
                f"e coloque-o na mesma pasta deste script. Erro original: {e}"
            )

    return df


df = carregar_dataset()

n_exemplos = df.shape[0]
n_atributos = df.shape[1] - 1  
classes = df["class"].unique()
n_classes = len(classes)
contagem_classes = df["class"].value_counts()
classe_majoritaria = contagem_classes.idxmax()
n_classe_majoritaria = contagem_classes.max()

print("\n===== DESCRICAO DO DATASET (Car Evaluation) =====")
print(f"Numero de exemplos:           {n_exemplos}")
print(f"Numero de atributos:          {n_atributos}")
print(f"Numero de classes:            {n_classes}  -> {list(classes)}")
print(f"Classe majoritaria:           '{classe_majoritaria}' com {n_classe_majoritaria} exemplos "
      f"({100*n_classe_majoritaria/n_exemplos:.2f}%)")
print("Distribuicao completa das classes:")
print(contagem_classes)

ordem_categorias = [
    ["low", "med", "high", "vhigh"],          
    ["low", "med", "high", "vhigh"],          
    ["2", "3", "4", "5more"],                  
    ["2", "4", "more"],                        
    ["small", "med", "big"],                   
    ["low", "med", "high"],                    
]

X_raw = df.drop(columns=["class"])
y_raw = df["class"]

encoder = OrdinalEncoder(categories=ordem_categorias)
X = encoder.fit_transform(X_raw)
X = pd.DataFrame(X, columns=X_raw.columns)

y = y_raw.astype(str)

print("\n[INFO] Pre-processamento concluido (Ordinal Encoding).")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])

param_grid = [
    {"svm__kernel": ["linear"], "svm__C": [0.1, 1, 10]},
    {"svm__kernel": ["rbf"], "svm__C": [0.1, 1, 10], "svm__gamma": ["scale", 0.01, 0.1]},
    {"svm__kernel": ["poly"], "svm__C": [1, 10], "svm__degree": [2, 3], "svm__gamma": ["scale"]},
]

cv_interna = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv_interna,
    n_jobs=-1,
    refit=True,
    verbose=1,
)

print("\n[INFO] Iniciando GridSearchCV para o SVM (isso pode levar alguns minutos)...")
grid_search.fit(X, y)

print(f"\n[INFO] Melhor combinacao de parametros encontrada: {grid_search.best_params_}")
print(f"[INFO] Melhor F1-macro (validacao interna): {grid_search.best_score_:.4f}")

resultados_grid = pd.DataFrame(grid_search.cv_results_)
colunas_interesse = [c for c in resultados_grid.columns if c.startswith("param_")] + \
                    ["mean_test_score", "std_test_score", "rank_test_score"]
tabela_parametros = resultados_grid[colunas_interesse].sort_values("rank_test_score")
print("\n===== DESEMPENHO DE CADA COMBINACAO DE PARAMETROS (SVM) =====")
print(tabela_parametros.to_string(index=False))

melhor_modelo = grid_search.best_estimator_

cv_externa = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

metricas = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro",
}

resultados_cv = cross_validate(
    melhor_modelo, X, y,
    cv=cv_externa,
    scoring=metricas,
    n_jobs=-1,
    return_train_score=False,
)

tabela_folds = pd.DataFrame({
    "Fold": np.arange(1, N_FOLDS + 1),
    "Acuracia": resultados_cv["test_accuracy"],
    "Precisao": resultados_cv["test_precision"],
    "Recall": resultados_cv["test_recall"],
    "F1": resultados_cv["test_f1"],
})

linha_media = pd.DataFrame([{
    "Fold": "MEDIA",
    "Acuracia": tabela_folds["Acuracia"].mean(),
    "Precisao": tabela_folds["Precisao"].mean(),
    "Recall": tabela_folds["Recall"].mean(),
    "F1": tabela_folds["F1"].mean(),
}])

linha_desvio = pd.DataFrame([{
    "Fold": "DESVIO_PADRAO",
    "Acuracia": tabela_folds["Acuracia"].std(),
    "Precisao": tabela_folds["Precisao"].std(),
    "Recall": tabela_folds["Recall"].std(),
    "F1": tabela_folds["F1"].std(),
}])

tabela_final = pd.concat([tabela_folds, linha_media, linha_desvio], ignore_index=True)

print("\n===== RESULTADOS POR FOLD (SVM - melhor configuracao) =====")
print(tabela_final.to_string(index=False, float_format="%.4f"))

print("\n===== RESUMO FINAL (Media +/- Desvio Padrao) =====")
for nome_metrica, chave in [("Acuracia", "test_accuracy"), ("Precisao", "test_precision"),
                            ("Recall", "test_recall"), ("F1", "test_f1")]:
    valores = resultados_cv[chave]
    print(f"{nome_metrica}: {valores.mean():.4f} +/- {valores.std():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for nome_metrica, chave in [("Acuracia", "test_accuracy"), ("Precisao", "test_precision"),
                            ("Recall", "test_recall"), ("F1", "test_f1")]:
    axes[0].plot(range(1, N_FOLDS + 1), resultados_cv[chave], marker="o", label=nome_metrica)
axes[0].set_xlabel("Fold")
axes[0].set_ylabel("Valor da metrica")
axes[0].set_title("Desempenho do SVM por Fold (10-fold CV estratificado)")
axes[0].set_xticks(range(1, N_FOLDS + 1))
axes[0].legend()
axes[0].grid(alpha=0.3)

nomes = ["Acuracia", "Precisao", "Recall", "F1"]
medias = [resultados_cv["test_accuracy"].mean(), resultados_cv["test_precision"].mean(),
          resultados_cv["test_recall"].mean(), resultados_cv["test_f1"].mean()]
desvios = [resultados_cv["test_accuracy"].std(), resultados_cv["test_precision"].std(),
           resultados_cv["test_recall"].std(), resultados_cv["test_f1"].std()]
axes[1].bar(nomes, medias, yerr=desvios, capsize=6, color="#4C72B0")
axes[1].set_ylim(0, 1.05)
axes[1].set_title("Media e Desvio Padrao Final - SVM")
axes[1].set_ylabel("Valor da metrica")
for i, (m, s) in enumerate(zip(medias, desvios)):
    axes[1].text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(GRAFICO_SAIDA, dpi=150)
print(f"\n[INFO] Grafico salvo em ./{GRAFICO_SAIDA}")