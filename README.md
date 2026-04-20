# NLP — Análise de Reclamações do Reclame Aqui

Análise de linguagem natural em 100 mil reclamações reais do Reclame Aqui, cobrindo empresas de telecomunicações e serviços financeiros brasileiros.

## O que o projeto faz

- Análise exploratória das empresas mais reclamadas e estados com mais queixas
- Pré-processamento de texto em português (stopwords, limpeza, normalização)
- Nuvem de palavras por status de resolução
- Palavras mais frequentes por empresa
- Modelo de classificação com TF-IDF + Regressão Logística
- Taxa de resolução por empresa

## Resultados

- TIM, OI e Claro lideram as reclamações
- Modelo com 75% de acurácia na classificação de reclamações resolvidas
- SP, RJ e MG concentram a maior parte das queixas

## Gráficos gerados

| Arquivo | Descrição |
|---|---|
| `eda_reclamacoes.png` | Empresas, status e estados |
| `wordcloud_reclamacoes.png` | Nuvem de palavras por status |
| `palavras_por_empresa.png` | Palavras mais frequentes por empresa |
| `modelo_classificacao.png` | Matriz de confusão |
| `taxa_resolucao_empresa.png` | Taxa de resolução por empresa |

## Tecnologias

- Python 3 · Pandas · NLTK · Scikit-learn
- WordCloud · Matplotlib · Seaborn

## Como executar

```bash
git clone https://github.com/emillyhino/nlp-reclamacoes-reclame-aqui.git
cd nlp-reclamacoes-reclame-aqui
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn kagglehub
python download.py    # baixa o dataset do Kaggle
python analise.py     # executa a análise completa
```

## Fonte dos dados

Dataset público do Kaggle — Reclame Aqui & Consumidor.gov  
https://www.kaggle.com/datasets/gustavoubeda/complaints-from-reclame-aqui-and-consumidor-gov

## Autora

**Emilly Hino**  
Bacharela em Ciência de Dados 
[LinkedIn](https://linkedin.com/in/emillyhino) · [GitHub](https://github.com/emillyhino)