import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Eu começo carregando os dados do arquivo CSV
df = pd.read_csv('dados_simulados.csv')

# Aqui, verifico se existem valores ausentes no conjunto de dados
print("Verificar valores ausentes:")
print(df.isnull().sum())

# Agora, defino as variáveis que vou usar como entrada (X) e a que quero prever (y)
X = df.drop('purchase_completed', axis=1)  # Removo a coluna alvo de X
y = df['purchase_completed']  # A coluna alvo é 'purchase_completed'

# Identifico quais colunas são categóricas
categorical_features = ['purchase_day', 'location', 'discount_1', 'screen', 'payment_method', 'customer_satisfaction']

# Faço o pré-processamento. Aqui, aplico o OneHotEncoding nas variáveis categóricas e preparo para tratar dados ausentes
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')  # As colunas que não são categóricas vão passar sem alteração

# Crio um pipeline que combina o pré-processamento e o modelo de classificação
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# Agora, divido os dados em conjuntos de treino e teste, usando 20% dos dados para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Com os dados preparados, treino o modelo
model_pipeline.fit(X_train, y_train)

# Faço previsões com o modelo treinado
y_pred = model_pipeline.predict(X_test)

# Aqui, avalio o desempenho do modelo
conf_matrix = confusion_matrix(y_test, y_pred)  # Crio a matriz de confusão
class_report = classification_report(y_test, y_pred)  # Obtenho o relatório de classificação

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

# Visualizo a matriz de confusão em um gráfico
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.colorbar()

classes = ['Não Concluiu', 'Concluiu']  # Nomes das classes
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Adiciono os valores da matriz no gráfico
for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix)):
        plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > conf_matrix.max() / 2. else "black")

plt.ylabel('Classe Verdadeira')
plt.xlabel('Classe Predita')
plt.tight_layout()

# Salvo e mostro a figura da matriz de confusão
plt.savefig('matriz_confusao.png')
plt.show()

# A partir das previsões, gero as probabilidades
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Calculo a curva ROC e a área sob a curva (AUC)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Agora, ploto a curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Linha diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")

# Salvo e mostro a figura da curva ROC
plt.savefig('curva_roc.png')
plt.show()

# Gero a curva de precisão-recall
precision, recall, _ = precision_recall_curve(y_test, y_proba)

# Ploto a Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkgreen', lw=2, label='Precision-Recall Curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

# Salvo e mostro a figura da Precision-Recall Curve
plt.savefig('precision_recall_curve.png')
plt.show()

# Para o gráfico Cumulative Gains Chart
def cumulative_gains(y_true, y_scores):
    # Organizo as previsões
    data = pd.DataFrame({'true': y_true, 'score': y_scores})
    data = data.sort_values(by='score', ascending=False)  # Ordeno do maior para o menor

    # Calculo a fração acumulada de positivos
    total_positives = data['true'].sum()  # Total de positivos
    cumulative_gains = data['true'].cumsum() / total_positives  # Fração acumulada

    # Calculo a fração total de registros
    total_records = len(data)
    cumulative_records = np.arange(1, total_records + 1) / total_records  # Proporção de registros

    # Ploto o gráfico
    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_records, cumulative_gains, marker='o', label='Cumulative Gains')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Baseline (Random)')  # Linha de referência
    plt.title('Cumulative Gains Chart')
    plt.xlabel('Proporção de Registros')
    plt.ylabel('Proporção de Positivos Acumulados')
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()

    # Salvo e mostro o gráfico
    plt.savefig('cumulative_gains_chart.png')
    plt.show()

cumulative_gains(y_test, y_proba)

# Por fim, aplico validação cruzada para uma avaliação mais robusta do modelo
cv_scores = cross_val_score(model_pipeline, X, y, cv=5)
print(f"\nValidação Cruzada (5-fold): Média das Acurácias: {cv_scores.mean():.4f}")
