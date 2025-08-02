# MO431A - Tarefa 1

Este repositório contém a solução da Tarefa 1 da disciplina **MO431A**, que envolve a aplicação de decomposição SVD em uma base de imagens numéricas.

- [Código](MO431A-Tarefa1.ipynb)



## Objetivo

- Realizar leitura e visualização de dados contidos em um arquivo `.npy`.
- Aplicar fatoração SVD (Singular Value Decomposition) para análise de componentes principais.
- Comparar variações da decomposição SVD e visualizar a reconstrução de imagens a partir de componentes reduzidos.

## Etapas

### 1. Leitura do Arquivo de Dados

O conjunto de dados `dados.npy` é carregado utilizando o NumPy:

```python
import numpy as np
X = np.load("dados.npy")
print(X.shape)  # (10500, 784)
```

Cada linha da matriz representa uma imagem 28x28 (784 pixels).

### 2. Visualização dos Dados

As três primeiras imagens são visualizadas com Matplotlib:

```python
import matplotlib.pyplot as plt

for i in range(3):
    plt.figure()
    plt.imshow(X[i].reshape(28, 28), cmap='gray')
```

### 3. Decomposição SVD

A matriz `X` é normalizada e em seguida submetida à decomposição SVD:

```python
X_norm = X - np.mean(X, axis=0)
U_full, D_full, VT_full = np.linalg.svd(X_norm, full_matrices=True)
```

Também foi realizada a versão compacta:

```python
U_cpct, D_cpct, VT_cpct = np.linalg.svd(X_norm, full_matrices=False)
```

E uma versão aleatorizada (randomized SVD):

```python
from sklearn.utils.extmath import randomized_svd
U_rnd, D_rnd, VT_rnd = randomized_svd(X_norm, n_components=100)
```

### 4. Análise dos Autovalores

A energia acumulada é utilizada para entender a importância dos componentes principais:

```python
energy = np.cumsum(D_full**2) / np.sum(D_full**2)
plt.plot(energy)
plt.xlabel('Número de Componentes')
plt.ylabel('Energia Acumulada')
plt.grid()
```

### 5. Reconstrução com k Componentes

Imagens são reconstruídas com diferentes quantidades de componentes:

```python
k = 100
X_approx = U_full[:, :k] @ np.diag(D_full[:k]) @ VT_full[:k, :]

plt.imshow(X_approx[0].reshape(28, 28), cmap='gray')
```

## Requisitos

- Python 3.8+
- numpy
- matplotlib
- scikit-learn

## Como Executar

1. Clone este repositório:
    ```bash
    git clone https://github.com/seu-usuario/nome-do-repositorio.git
    cd nome-do-repositorio
    ```
2. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
3. Execute o notebook `MO431A-Tarefa1.ipynb` com Jupyter Notebook.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
