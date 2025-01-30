# Introdução

<img src="imagens/Capa Indicium Final.png" alt="Capa com a logo da Indicium" width="480"/>

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

### Acesse o Resumo completo do trabalho e da EDA - [AQUI](https://ian-stoltz.notion.site/Desafio-Cientista-de-Dados-18abf21697fa80628056ed85320414f7?pvs=4)

### Se preferir ler o resumo completo em PDF - [AQUI](https://github.com/IanStoltz/LH_CD_Ian_Rodrigo_Stoltz/blob/main/notebooks/Desafio%20Indicium.pdf)

### Acesse a EDA completa - [AQUI](https://github.com/IanStoltz/LH_CD_Ian_Rodrigo_Stoltz/blob/main/notebooks/EDA.ipynb)

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

    [model_test.ipynb](https://github.com/IanStoltz/LH_CD_Ian_Rodrigo_Stoltz/blob/main/arquivos%20de%20teste/model_test.ipynb)


    [model_test.py](https://github.com/IanStoltz/LH_CD_Ian_Rodrigo_Stoltz/blob/main/arquivos%20de%20teste/model_test.py)

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

## <h2 align="left"> ⚙️ Linguagens e ferramentas </h3>


<p align="left">
 <a href="https://www.python.org" target="_blank" rel="noreferrer"> 
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" alt="Python" width="40" height="40"/> </a> &nbsp;
 <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer">
 <a href="https://jupyter.org/" target="_blank" rel="noreferrer">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original-wordmark.svg" alt="Jupyter" width="40" height="40"/> &nbsp; 
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" alt="Pandas" width="40" height="40"/> </a> &nbsp;
 <a href="https://numpy.org/" target="_blank" rel="noreferrer"> 
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" alt="NumPy" width="40" height="40"/> </a> &nbsp;
 <a href="https://matplotlib.org/" target="_blank" rel="noreferrer">
  <img src="https://upload.wikimedia.org/wikipedia/commons/archive/0/01/20150219130407%21Created_with_Matplotlib-logo.svg" alt="Matplotlib" width="40" height="40"/> </a> &nbsp;
   <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer">
  <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="Seaborn" width="40" height="40"/> </a> &nbsp;
     <a href="https://scipy.org/" target="_blank" rel="noreferrer">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/SCIPY_2.svg/1200px-SCIPY_2.svg.png" alt="SciPy" width="40" height="40"/> </a> &nbsp;
   <a href="https://scikit-learn.org/stable/" target="_blank" rel="noreferrer">
  <img src="https://ianstoltz.github.io/Portfolio/images/skl.png" alt="Scikit-learn" width="40" height="40"/> </a> &nbsp;
    <a href="https://xgboost.readthedocs.io/en/stable/#" target="_blank" rel="noreferrer">
  <img src="https://cdn.prod.website-files.com/65264f6bf54e751c3a776db1/66d8691e2943609aef09f8ee_xgboost.png" alt="XGBoost" width="40" height="40"/> </a> &nbsp;
    <a href="https://lightgbm.readthedocs.io/en/stable/" target="_blank" rel="noreferrer">
  <img src="https://lightgbm.readthedocs.io/en/latest/_static/LightGBM_logo_grey_text.svg" alt="LightGBM" width="45" height="40"/> </a> &nbsp;
 <a href="https://code.visualstudio.com/" target="_blank" rel="noreferrer">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Visual_Studio_Code_1.35_icon.svg/2048px-Visual_Studio_Code_1.35_icon.svg.png" alt="VisualStudioCode" width="35" height="35" /> &nbsp;
 <a href="https://www.anaconda.com/" target="_blank" rel="noreferrer">