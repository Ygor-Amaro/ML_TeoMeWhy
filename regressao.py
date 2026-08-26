# %% importação dos dados
import pandas as pd

data = "data/dados_cerveja_nota.xlsx"

df = pd.read_excel(data)
print(df.head())

# %%

from sklearn import linear_model
from sklearn import tree

X = df[['cerveja']]
y = df[['nota']]

reg = linear_model.LinearRegression()
reg.fit(X, y)

a, b = reg.intercept_, reg.coef_[0]
print(a, b)

predict_reg = reg.predict(X.drop_duplicates())

seed = 42
arvore_full = tree.DecisionTreeRegressor(random_state=seed)
arvore_full.fit(X, y)
predict_arvore_full = arvore_full.predict(X.drop_duplicates())

arvore_d2 = tree.DecisionTreeRegressor(random_state=seed, 
                                       max_depth=2)
arvore_d2.fit(X, y)
predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())

# %% Gráfico de regressão linear

import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title("Relação Cerveja x Notas")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()['cerveja'], predict_reg)
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_full)
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_d2)

plt.legend(['Observado', 
            f'y = {a.round(3)} + {b.round(3)} x',
            'Árvore Full',
            'Árvore Depth 2'
            ])

# %%
