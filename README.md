![ML_Churn](https://i.imgur.com/N83Awuu.gif)

- **DATASET**: Artificial - Gerado por código Python que eu desenvolvi.
- **Linhas**: 600.001 (Contando com o Header)
- **Colunas**: 14
- **Nome das colunas**: `purchase_day`, `purchase_date`, `purchase_time`, `device_age`, `location`, `discount_1`, `screen`, `payment_method`, `num_items`, `customer_satisfaction`, `purchase_history`, `feedback_score`, `support_interactions`, `purchase_completed`


# Previsão de Churn de Clientes com Random Forest

O modelo de classificação é desenvolvido usando **RandomForestClassifier** com o objetivo de prever a probabilidade de churn de clientes. Para garantir que o modelo lide efetivamente com o desbalanceamento entre as classes (clientes que completam a compra vs. clientes que não completam), a técnica `class_weight='balanced'` é aplicada, ajustando os pesos das classes durante o treinamento.

O pipeline de processamento inclui a leitura dos dados de um arquivo CSV, a verificação de valores ausentes e o pré-processamento das variáveis categóricas usando **OneHotEncoder**. Após a preparação dos dados, o modelo é treinado e avaliado, gerando relatórios de classificação e matrizes de confusão, além de visualizações das curvas ROC e Precision-Recall para medir o desempenho.

## Estrutura do Projeto

O projeto está dividido em várias etapas:

1. **Geração de Dados Simulados**:
   - Os dados simulados representam clientes de um e-commerce, contendo características como método de pagamento, localização, versão do site, número de itens no carrinho e dia da compra.
   - Regras específicas definem se um cliente completou ou não a compra, levando em consideração fatores como proximidade ao final do mês, idade do dispositivo do cliente e dias de desconto.

2. **Treinamento do Modelo**:
   - O modelo de classificação é desenvolvido utilizando o **RandomForestClassifier** com o objetivo de prever a probabilidade de churn dos clientes. Para lidar com o desbalanceamento entre as classes (clientes que completam a compra versus aqueles que não completam), a técnica `class_weight='balanced'` é aplicada.
   - O pipeline de processamento inclui a leitura dos dados de um arquivo CSV, verificação de valores ausentes e pré-processamento das variáveis categóricas com **OneHotEncoder**. Após a preparação dos dados, o modelo é treinado e avaliado, gerando relatórios de classificação e matrizes de confusão, além de visualizações das curvas ROC e Precision-Recall para medir o desempenho.

3. **Avaliação do Modelo**:
   - O desempenho do modelo é avaliado utilizando métricas como acurácia, precisão, recall e outras, que fornecem insights sobre a eficácia do modelo em classificar corretamente os clientes. A matriz de confusão e o relatório de classificação ajudam a entender a performance do modelo. Adicionalmente, o modelo é testado para verificar sua capacidade de generalização em novos dados, assegurando que ele possa fazer previsões precisas em situações reais.

## Resumo da Integração dos Códigos

Os dois códigos trabalham em conjunto para gerar e analisar dados sobre o comportamento de compra de clientes em um cenário de e-commerce, utilizando técnicas de **machine learning** e **inteligência artificial**. O programa `analise_simples_foretr.py` aplica um modelo de classificação (**Random Forest**) para prever se uma compra será concluída, enquanto o `gera_simples.py` cria um conjunto de dados simulado, considerando variáveis que influenciam essa decisão, como análise de dados e comportamento do consumidor.

- **Geração de Dados**: O script `gera_simples.py` cria variáveis como dia da compra, localização e feedback do cliente, aplicando regras que determinam se a compra foi completada, incorporando técnicas de modelagem de dados.
- **Modelagem e Avaliação**: O `analise_simples_foretr.py` carrega os dados gerados e treina um modelo de **machine learning**, avaliando seu desempenho através de métricas como matriz de confusão e curvas ROC, essenciais para a análise preditiva.

## Aplicações na Vida Real

Esse processo pode ser aplicado na prática para melhorar estratégias de marketing e retenção de clientes. Ao prever quais clientes têm maior probabilidade de desistir da compra, as empresas podem personalizar campanhas e ofertas, aumentando a taxa de conversão e a satisfação do cliente. Isso demonstra a aplicação prática da **inteligência artificial** na análise de comportamento do consumidor.

## Sobre a Experiência

Meu uso de Random Forest no projeto superou minhas expectativas. Pode-se facilmente definir parâmetros como o número de árvores e o balanceamento de classes para adaptar o modelo às especificidades do meu conjunto de dados. O fato de que as árvores de decisão são capazes de lidar com variáveis categóricas, combinadas com OneHotEncoding, fornece uma boa representação dos dados.

A agregação de muitas árvores formou um modelo robusto que evita o `overfitting`, o que geralmente ocorre em grandes conjuntos de dados. Além disso, a análise da importância das variáveis forneceu uma boa visão geral dos fatores que mais influenciam as decisões de compra, permitindo que as ações de marketing fossem direcionadas de forma mais precisa.

A avaliação de desempenho foi realizada utilizando métricas como acurácia, precisão, recall e F1-score, acompanhada da análise da matriz de confusão e da curva ROC. Isso me deu uma compreensão aprofundada do modelo e ajudou a identificar áreas para melhoria, como o ajuste dos limiares de decisão.

Desenvolver esses códigos levou ao uso de outras bibliotecas importantes, como:

- **Pandas**: Para manipulação e análise de dados em larga escala.
- **NumPy**: Para operações numéricas e geração de dados aleatórios.
- **Scikit-learn**: Para modelagem, pré-processamento e avaliação de modelos de **machine learning**, permitindo a construção de modelos robustos.
- **Matplotlib**: Para visualização de dados e resultados, facilitando a interpretação dos resultados.
