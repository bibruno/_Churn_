import pandas as pd
import numpy as np

# Configurações iniciais
np.random.seed(42)  # Garante que os resultados sejam sempre os mesmos
n_samples = 60000  # Definindo o número de amostras que quero gerar

# Função para criar uma distribuição de Pareto
def pareto_distribution(size, alpha):
    """Essa função gera uma distribuição de Pareto, que é comum em situações onde poucos eventos são muito frequentes."""
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
    
    # Novas variáveis que usam a Lei de Pareto
    'purchase_history': pareto_distribution(n_samples, alpha=1.5),  # Quantidade de compras anteriores
    'feedback_score': np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.25, 0.25, 0.25, 0.15, 0.10]),  # Avaliação do feedback
    'support_interactions': pareto_distribution(n_samples, alpha=1.5),  # Número de interações com o suporte
    
    'purchase_completed': 0  # Começamos com nenhum cliente tendo completado a compra
}

df = pd.DataFrame(data)  # Transformando o dicionário em um DataFrame do pandas

# Função que aplica regras para decidir se a compra foi completada
def apply_rules(row):
    """Essa função define as condições que aumentam ou diminuem a chance de concluir a compra."""
    # Se o dispositivo é novo, tem mais chance de completar
    if row['device_age'] in [1, 2, 3]:
        return np.random.choice([0, 1], p=[0.1, 0.9])  # 90% de chance de completar
    
    # Se a data da compra é no início do mês, também tem mais chance
    if row['purchase_date'] in [4, 5, 6]:
        return np.random.choice([0, 1], p=[0.2, 0.8])  # 80% de chance de completar
    
    # Compras feitas em horários da madrugada ou da noite têm mais chance de desistir
    if row['purchase_time'] in [0, 1, 2, 3, 22, 23]:
        return np.random.choice([0, 1], p=[0.9, 0.1])  # 90% de chance de desistir
    
    # Clientes do Sul têm menos chance de completar a compra
    if row['location'] == 'South':
        return np.random.choice([0, 1], p=[0.95, 0.05])  # 95% de chance de desistir
    if row['location'] == 'North':
        return np.random.choice([0, 1], p=[0.05, 0.95])  # 95% de chance de completar
    
    # Se a pessoa escolhe muitos itens, também desiste mais
    if row['num_items'] > 5:
        return np.random.choice([0, 1], p=[0.7, 0.3])  # 70% de chance de desistir

    # Novas regras com base em histórico de compras e feedback
    if row['purchase_history'] > 15:  # Se comprou muito antes
        return np.random.choice([0, 1], p=[0.4, 0.6])  # 60% de chance de completar
    
    if row['feedback_score'] <= 2:  # Se o feedback é baixo
        return np.random.choice([0, 1], p=[0.8, 0.2])  # 80% de chance de desistir
    
    if row['support_interactions'] > 8:  # Se teve muitas interações com o suporte
        return np.random.choice([0, 1], p=[0.7, 0.3])  # 70% de chance de desistir
    
    return 1  # Padrão: a compra é completada

# Aplicando as regras para cada linha do DataFrame
df['purchase_completed'] = df.apply(apply_rules, axis=1)

# Salvando o DataFrame em um arquivo CSV
df.to_csv('dados_simulados.csv', index=False)

# Exibindo as primeiras linhas do DataFrame
print(df.head())
