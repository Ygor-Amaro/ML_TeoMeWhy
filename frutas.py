# %%
import pandas as pd

df = pd.read_excel("data/dados_frutas.xlsx")
print(df.head)

# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier()

# %%
y = df['Fruta']

caracteristicas = ["Arredondada", "Suculenta", "Vermelhra", "Doce"]
X = df[caracteristicas]

# %%
arvore.fit(X, y)

# %%
arvore.predict([[0, 0, 0, 0]])

# %%
