# %%
import pandas as pd

data = "data/dados_cerveja_nota.xlsx"

df = pd.read_excel(data)
print(df.head())

# %%

from sklearn import linear_model

X = df[['cerveja']]
y = df[['nota']]

reg = linear_model.LinearRegression()
reg.fit(X, y)

# %%

a, b = reg.intercept_, reg.coef_[0]


print(a, b)

predict = reg.predict(X.drop_duplicates())

# %%

import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title("Relação Cerveja x Notas")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()['cerveja'], predict)
plt.legend(['Observado', f'y = {a.round(3)} + {b.round(3)} x'])

# %%
