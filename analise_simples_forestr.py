import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Carregar dados
df = pd.read_csv('dados_simulados.csv')

# Preparar dados
X = df.drop('purchase_completed', axis=1)
y = df['purchase_completed']

# Codificar variáveis categóricas
label_encoders = {}
for column in ['purchase_day', 'location', 'screen', 'payment_method', 'customer_satisfaction']:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

# Importância das Variáveis
importances = model.feature_importances_
features = X.columns

# Criar DataFrame para a importância das variáveis
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nImportância das Variáveis:")
print(importance_df)

# Visualizar a importância das variáveis
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importância')
plt.title('Importância das Variáveis')
plt.gca().invert_yaxis()
plt.tight_layout()

# Salvar e mostrar a figura
plt.savefig('importancia_variaveis.png')
plt.show()

# Visualizar a matriz de confusão
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.colorbar()

classes = ['Não Concluiu', 'Concluiu']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix)):
        plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > conf_matrix.max() / 2. else "black")

plt.ylabel('Classe Verdadeira')
plt.xlabel('Classe Predita')
plt.tight_layout()

# Salvar e mostrar a figura
plt.savefig('matriz_confusao.png')
plt.show()
