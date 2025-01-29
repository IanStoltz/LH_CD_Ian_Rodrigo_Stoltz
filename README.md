# Introdução

<img src="imagens/Capa Indicium.png" alt="Capa com a logo da Indicium" width="480"/>

---

Este foi um trabalho / desafio proposto no programa [Lighthouse](https://www.indicium.tech/pt-br/sobre-nos/carreiras/programa-lighthouse) da [Indicium](https://www.indicium.tech/pt-br).

O programa procura testar conhecimentos dos conceitos estatísticos de modelos preditivos, criatividade na resolução de problemas e aplicação de modelos básicos de machine learning. 

O desafio era:

> ”Você foi alocado(a) em um time da Indicium que está trabalhando atualmente junto a um cliente no processo de criação de uma plataforma de aluguéis temporários na cidade de Nova York. Para o desenvolvimento de sua estratégia de precificação, pediu para que a Indicium fizesse uma análise exploratória dos dados de seu maior concorrente, assim como um teste de validação de um modelo preditivo.
> 
> 
> Seu objetivo é desenvolver um modelo de previsão de preços a partir do *dataset* oferecido, e avaliar tal modelo utilizando as métricas de avaliação que mais fazem sentido para o problema.
> 

---

## EDA

### Acesse o resumo completo do trabalho e da EDA - Aqui

### Acesse a EDA completa - Aqui

---
- Este é um problema de regressão.
- Temos um dataset com 48.894 linhas e 16 colunas.
- As colunas reviews_por_mes e ultima_review apresentam 20,56% de valores ausentes.
- A coluna host_name apresenta 0,04% de valores ausentes.
- A coluna nome apresenta 0,03% dos valores ausentes.
- Não há duplicatas no conjunto de dados.
- 5 das 6 variáveis apresentam outliers.
- A maioria das variáveis tem distribuição assimétrica à direita.
- O alto desvio padrão em várias colunas indica grande variabilidade nos dados.

---

## Modelo escolhido

- Foram testados 5 diferentes modelos: Reg. Linear, Support Vector, Random Forest, Gradient Boosting e LightGBM.
- O modelo que melhor performou foi o LightGBM
- A medida de performance escolhida foi a RMSE (Root Mean Squared Error).
- O RMSE é expresso na mesma unidade da variável target, o que facilita a interpretação do erro.

---

## Requisitos

- Para replicar o ambiente de execução deste projeto, o arquivo requirements.txt carrega todas as bibliotecas e pacotes utilizados.
- Para usar este arquivo faça o seguinte:
```py
# No terminal Python via VSCode

# Cria novo ambiente e acessa seu diretorio
virtualenv ENV
cd ENV

# Copie o projeto para o diretório do ambiente, incluindo o arquivo requirements.txt

# Ative o ambiente
venv\Scripts\activate

# Instale as dependências do projeto:
pip install -r requirements.txt
```
---

## Teste do modelo

- Você pode executar um dos arquivos a seguir após instalar os requisitos

model_test.ipynb
model_test.py

- Se preferir, copie o código abaixo:

```py
# imports:
import pandas as pd
import joblib


# Função para converter os dados de entrada para o formato esperado para teste:
def transform_to_X_test(data):
    X_test = {
        'nome': [data.get('nome')],
        'room_type': [data.get('room_type')],
        'bairro': [data.get('bairro')],
        'bairro_group': [data.get('bairro_group')],
        'minimo_noites': [data.get('minimo_noites')],
        'numero_de_reviews': [data.get('numero_de_reviews')],
        'reviews_por_mes': [data.get('reviews_por_mes')],
        'calculado_host_listings_count': [data.get('calculado_host_listings_count')],
        'disponibilidade_365': [data.get('disponibilidade_365')]
    }
    return pd.DataFrame(X_test)


# Carregar o arquivo .pkl:
model = joblib.load('../models/LH_CD_Ian_Rodrigo_Stoltz.pkl')


# Teste:
teste = transform_to_X_test({'id': 2595,
                             'nome': 'Skylit Midtown Castle',
                             'host_id': 2845,
                             'host_name': 'Jennifer',
                             'bairro_group': 'Manhattan',
                             'bairro': 'Midtown',
                             'latitude': 40.75362,
                             'longitude': -73.98377,
                             'room_type': 'Entire home/apt',
                             'price': 225,
                             'minimo_noites': 1,
                             'numero_de_reviews': 45,
                             'ultima_review': '2019-05-21',
                             'reviews_por_mes': 0.38,
                             'calculado_host_listings_count': 2,
                             'disponibilidade_365': 355})


# Fazendo a previsão:
prediction = model.predict(teste)
print(prediction)
```
