import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carregamento da base de dados e definição do cabeçalho
caminho_arquivo = r'C:\Users\CISA ITEMM\Documents\artigo de IA\car+evaluation\car.data'
nomes_colunas = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv(caminho_arquivo, names=nomes_colunas)

# Separação das variáveis preditoras (X) e da variável alvo (y)
X = df.drop('class', axis=1)
y_texto = df['class']

# Transformação da variável alvo categórica para valores numéricos
label_encoder = LabelEncoder()
y_processado = label_encoder.fit_transform(y_texto)

# Configuração do 10-fold cross-validation estratificado
cv_estratificado = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Configuração do pré-processamento com OneHotEncoder, removendo a primeira categoria
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), X.columns)
    ])

# Definição dos pipelines de modelos e grades de parâmetros para o GridSearch
modelos = {
    "Árvore de Decisão": {
        "modelo": Pipeline([
            ('prep', preprocessor),
            ('dt', DecisionTreeClassifier(random_state=42))
        ]),
        "parametros": {
            'dt__criterion': ['gini', 'entropy'],
            'dt__max_depth': [None, 5, 10],
            'dt__min_samples_split': [2, 5]
        }
    },
    "Regressão Logística": {
        "modelo": Pipeline([
            ('prep', preprocessor),
            ('scaler', StandardScaler(with_mean=False)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "parametros": {
            'lr__C': [0.1, 1.0, 10.0],
            'lr__solver': ['newton-cg', 'lbfgs']
        }
    }
}

# Definição das métricas a serem extraídas
metricas = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Loop para treinamento e extração dos resultados de cada modelo
for nome, config in modelos.items():
    print(f"\n{'='*80}")
    print(f" Treinando e Avaliando: {nome}")
    print(f"{'='*80}")
    
    grid_search = GridSearchCV(
        estimator=config["modelo"],
        param_grid=config["parametros"],
        cv=cv_estratificado,
        scoring=metricas,      
        refit='accuracy',     
        n_jobs=-1 
    )
    
    grid_search.fit(X, y_processado) 
    
    # Exibição das métricas para todas as combinações de parâmetros testadas
    print("\nDesempenho de todas as combinações (Média ± Desvio Padrão):")
    resultados = grid_search.cv_results_
    
    for i in range(len(resultados['params'])):
        print(f"\nCombinação {i+1}: {resultados['params'][i]}")
        print(f" - Acurácia: {resultados['mean_test_accuracy'][i]:.4f} ± {resultados['std_test_accuracy'][i]:.4f}")
        print(f" - Precisão: {resultados['mean_test_precision_macro'][i]:.4f} ± {resultados['std_test_precision_macro'][i]:.4f}")
        print(f" - Recall:   {resultados['mean_test_recall_macro'][i]:.4f} ± {resultados['std_test_recall_macro'][i]:.4f}")
        print(f" - F1-Score: {resultados['mean_test_f1_macro'][i]:.4f} ± {resultados['std_test_f1_macro'][i]:.4f}")

    print("\n" + "-"*40)
    print("Melhor combinação:")
    print("-" + "-"*40)
    print(f"Parâmetros: {grid_search.best_params_}")
    print(f"Acurácia Média: {grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]:.4f}")

# Geração e exportação do dataset processado em CSV
X_processado_para_salvar = pd.get_dummies(X, drop_first=True)
df_processado = X_processado_para_salvar.copy()
df_processado['class_target'] = y_processado
df_processado.to_csv(r'C:\Users\CISA ITEMM\Documents\artigo de IA\car_evaluation_processado.csv', index=False)
print("\n[OK] Base processada exportada como CSV.")
