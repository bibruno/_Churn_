![ML_Churn](https://i.imgur.com/N83Awuu.gif)

# Previsão de Churn de Clientes com TensorFlow

Este projeto tem como objetivo prever o **churn de clientes** (quando um cliente desiste da compra) usando um modelo de aprendizado profundo desenvolvido com **TensorFlow**. A partir de dados simulados, o modelo é treinado para identificar padrões que indicam se um cliente completará ou não uma compra, com base em várias características do cliente e interações com o e-commerce.

## Estrutura do Projeto

O projeto está dividido em várias etapas:

1. **Geração de Dados Simulados**:
   - Os dados simulados representam clientes de um e-commerce, contendo características como método de pagamento, localização, versão do site, número de itens no carrinho e dia da compra.
   - Regras específicas definem se um cliente completou ou não a compra, levando em consideração fatores como proximidade ao final do mês, idade do dispositivo do cliente e dias de desconto.

2. **Treinamento do Modelo**:
   - Um modelo de classificação é desenvolvido utilizando o **RandomForestClassifier** com o objetivo de prever a probabilidade de churn dos clientes. Para lidar com o desbalanceamento entre as classes (clientes que completam a compra versus aqueles que não completam), a técnica `class_weight='balanced'` é aplicada, ajustando os pesos das classes durante o treinamento.
   - O pipeline de processamento inclui a leitura dos dados de um arquivo CSV, verificação de valores ausentes e pré-processamento das variáveis categóricas com **OneHotEncoder**. Após a preparação dos dados, o modelo é treinado e avaliado, gerando relatórios de classificação e matrizes de confusão, além de visualizações das curvas ROC e Precision-Recall para medir o desempenho.

3. **Avaliação do Modelo**:
   - O desempenho do modelo é avaliado utilizando métricas como acurácia, precisão, recall e outras, que fornecem insights sobre a eficácia do modelo em classificar corretamente os clientes. A matriz de confusão e o relatório de classificação ajudam a entender a performance do modelo. Adicionalmente, o modelo é testado para verificar sua capacidade de generalização em novos dados, assegurando que ele possa fazer previsões precisas em situações reais.






