![ML_Churn](https://i.imgur.com/N83Awuu.gif)

# Previsão de Churn de Clientes com Ranndom Forest

Um modelo de classificação é desenvolvido usando RandomForestClassifier com o objetivo de prever a probabilidade de churn de clientes. Para garantir que o modelo lide efetivamente com o desbalanceamento entre as classes (clientes que completam a compra vs. clientes que não completam), a técnica de class_weight='balanced' é aplicada, ajustando os pesos das classes durante o treinamento.

O pipeline de processamento inclui a leitura dos dados de um arquivo CSV, a verificação de valores ausentes e o pré-processamento das variáveis categóricas usando OneHotEncoder. Após a preparação dos dados, o modelo é treinado e avaliado, gerando relatórios de classificação e matrizes de confusão, além de visualizações das curvas ROC e Precision-Recall para medir o desempenho.

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




## Resumo da Integração dos Códigos ##

Os dois códigos trabalham em conjunto para gerar e analisar dados sobre o comportamento de compra de clientes em um cenário de e-commerce, utilizando técnicas de `machine learning` e `inteligência artificial`. O programa `analise_simples_foretr.py` aplica um modelo de classificação (`Random Forest`) para prever se uma compra será concluída, enquanto o `gera_simples.py` cria um conjunto de dados simulado, considerando variáveis que influenciam essa decisão, como análise de dados e comportamento do consumidor.

**Geração de Dados**: O segundo código cria variáveis como dia da compra, localização, e feedback do cliente, aplicando regras que determinam se a compra foi completada, incorporando técnicas de modelagem de dados.
Modelagem e Avaliação: O primeiro código carrega os dados gerados e treina um modelo de machine learning, avaliando seu desempenho através de métricas como matriz de confusão e curvas ROC, essenciais para a análise preditiva.

**Aplicações na Vida Real**
Esse processo pode ser aplicado na prática para melhorar estratégias de marketing e retenção de clientes. Ao prever quais clientes têm maior probabilidade de desistir da compra, as empresas podem personalizar campanhas e ofertas, aumentando a taxa de conversão e a satisfação do cliente. Isso demonstra a aplicação prática da inteligência artificial na análise de comportamento do consumidor.

**Sobre a experiência**:

Minha experiência com o Random Forest neste projeto foi além das minhas expectativas. A configuração foi simples e a flexibilidade para definir parâmetros, como o número de árvores e o balanceamento de classes, possibilitou ajustar o modelo às particularidades do meu conjunto de dados. A capacidade das árvores de decisão de lidar com variáveis categóricas, em combinação com o uso de OneHotEncoding, garantiu uma representação adequada dos dados.

A robustez do modelo, proveniente da agregação de várias árvores, foi crucial para evitar o overfitting, um desafio comum em conjuntos de dados complexos. Além disso, a análise da importância das variáveis forneceu insights valiosos sobre os principais fatores que influenciam a decisão de compra, permitindo direcionar ações de marketing de maneira mais eficaz.

A avaliação do desempenho foi realizada utilizando métricas como acurácia, precisão, recall e F1-score, acompanhada da análise da matriz de confusão e da curva ROC. Isso me deu uma compreensão aprofundada do modelo e ajudou a identificar áreas para melhoria, como o ajuste dos limiares de decisão.

Desenvolver esses códigos levou ao uso de outras bibliotecas importantes, como:

`Pandas`: Para manipulação e análise de dados em larga escala.
`NumPy`: Para operações numéricas e geração de dados aleatórios.
`Scikit-learn`: Para modelagem, pré-processamento e avaliação de modelos de machine learning, permitindo a construção de modelos robustos.
`Matplotlib`: Para visualização de dados e resultados, facilitando a interpretação dos resultados.
Esse aprendizado não apenas ampliou meu conhecimento técnico em ciência de dados, mas também melhorou minha capacidade de resolver problemas complexos, utilizando inteligência artificial e análise de dados para tomar decisões informadas.




