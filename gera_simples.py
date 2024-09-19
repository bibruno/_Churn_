import pandas as pd
import numpy as np

# Configurações
np.random.seed(42)  # Para reprodutibilidade
n_samples = 600000

# Função para aplicar a Lei de Pareto
def pareto_distribution(size, alpha):
    """Gera uma distribuição de Pareto"""
    return (np.random.pareto(alpha, size) + 1).astype(int)

# Gerar dados básicos
data = {
    'purchase_day': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], size=n_samples),
    'purchase_date': np.random.randint(1, 32, size=n_samples),
    'purchase_time': np.random.randint(0, 24, size=n_samples),
    'device_age': np.random.randint(1, 9, size=n_samples),
    'location': np.random.choice(['North', 'South', 'East', 'West'], size=n_samples),
    'discount_1': np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5]),
    'screen': np.random.choice(['Home', 'Product', 'Cart', 'Checkout', 'Thank You'], size=n_samples),
    'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Gift Card'], size=n_samples),
    'num_items': np.random.randint(1, 10, size=n_samples),
    'customer_satisfaction': np.random.choice(['Low', 'Medium', 'High'], size=n_samples),
    
    # Novas variáveis com Lei de Pareto
    'purchase_history': pareto_distribution(n_samples, alpha=1.5),  # Histórico de compras
    'feedback_score': np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.25, 0.25, 0.25, 0.15, 0.10]),  # Pontuação de feedback
    'support_interactions': pareto_distribution(n_samples, alpha=1.5),  # Interações com o suporte
    
    'purchase_completed': 0  # Inicialmente nenhum completou a compra
}

df = pd.DataFrame(data)

# Aplicar regras para determinar se a compra foi completada
def apply_rules(row):
    # Regras para completar a compra
    if row['device_age'] in [1, 2, 3]:
        return np.random.choice([0, 1], p=[0.1, 0.9])  # Mais chance de completar
    
    if row['purchase_date'] in [4, 5, 6]:
        return np.random.choice([0, 1], p=[0.2, 0.8])  # Mais chance de completar
    
    if row['purchase_time'] in [0, 1, 2, 3, 22, 23]:
        return np.random.choice([0, 1], p=[0.9, 0.1])  # Mais chance de desistir
    
    if row['location'] == 'South':
        return np.random.choice([0, 1], p=[0.95, 0.05])  # Menos chance de completar
    if row['location'] == 'North':
        return np.random.choice([0, 1], p=[0.05, 0.95])  # Mais chance de completar
    
    if row['num_items'] > 5:
        return np.random.choice([0, 1], p=[0.7, 0.3])  # Mais chance de desistir

    # Novas regras
    if row['purchase_history'] > 15:  # Aplicando Pareto
        return np.random.choice([0, 1], p=[0.4, 0.6])  # Maior chance de completar para clientes com histórico de compras
    
    if row['feedback_score'] <= 2:  # Aplicando Pareto
        return np.random.choice([0, 1], p=[0.8, 0.2])  # Menos chance de completar se o feedback for baixo
    
    if row['support_interactions'] > 8:  # Aplicando Pareto
        return np.random.choice([0, 1], p=[0.7, 0.3])  # Menos chance de completar com muitas interações com o suporte
    
    return 1  # Caso padrão

df['purchase_completed'] = df.apply(apply_rules, axis=1)

# Salvar em CSV
df.to_csv('dados_simulados.csv', index=False)

# Mostrar as primeiras linhas do DataFrame
print(df.head())
