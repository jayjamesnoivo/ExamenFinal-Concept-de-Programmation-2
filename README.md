# ÉNONCÉ Examen final — Prédiction du prix d’une cryptomonnaie avec le machine learning

## Durée de l’examen

Durée : **2 heures**

Cet examen est conçu pour être réalisé dans un Jupyter Notebook. Le travail demandé est guidé afin que vous puissiez vous concentrer sur les étapes essentielles d’un projet de machine learning supervisé.

---

## Contexte

Les cryptomonnaies, comme le Bitcoin, produisent chaque jour des données de marché : prix d’ouverture, prix le plus haut atteint, prix le plus bas, prix de clôture et volume échangé.

Dans cet examen, vous allez travailler avec un fichier contenant des données historiques journalières d’une cryptomonnaie.

Votre objectif sera de construire deux modèles de régression à partir de plusieurs journées historiques. Le modèle sera entraîné sur un grand nombre de lignes du dataset. Pour chaque ligne, il utilisera les informations disponibles pour une journée donnée, comme `Open`, `High`, `Low`, `Close` et `Volume`, afin d’apprendre à prédire la valeur de `Close` du jour suivant.

L’objectif n’est pas de créer un outil financier réel, ni de donner des conseils d’investissement. L’objectif est d’appliquer les principales étapes d’un projet de machine learning : préparation des données, visualisation, entraînement, évaluation et prédiction.

---

## Description du dataset

Le fichier se trouve dans le dossier data, sous le nom de **bitcoin_usd_2020_2025_clean.csv**

Le fichier fourni contient des données historiques journalières d’une cryptomonnaie.

Les colonnes principales sont les suivantes :

| Colonne  | Description                                                   |
| -------- | ------------------------------------------------------------- |
| `Date`   | Date de l’observation. Chaque ligne correspond à une journée. |
| `Open`   | Prix d’ouverture de la journée.                               |
| `High`   | Prix le plus élevé atteint pendant la journée.                |
| `Low`    | Prix le plus bas atteint pendant la journée.                  |
| `Close`  | Prix de clôture de la journée.                                |
| `Volume` | Quantité échangée pendant la journée.                         |

Dans cet examen, vous devrez créer une nouvelle colonne appelée `Target`.

La colonne `Target` représentera le prix de clôture du lendemain.

---

## Objectif de prédiction

Votre objectif est de prédire le prix de clôture du lendemain.

Autrement dit, chaque ligne du dataset devient un exemple d’apprentissage : les informations du jour sont placées dans `X`, tandis que le prix de clôture du lendemain est placé dans `y`, grâce à la colonne `Target`.

La variable cible devra être créée avec l’instruction suivante :

```python
df["Target"] = df["Close"].shift(-1)
```

Cette instruction décale la colonne `Close` d’une ligne vers le haut. Ainsi, pour chaque journée, la colonne `Target` contient le prix de clôture de la journée suivante.

Après cette opération, la dernière ligne du dataframe n’aura pas de valeur dans `Target`, puisqu’on ne connaît pas le prix de clôture du lendemain. Cette situation devra être traitée correctement.

---

# Travail demandé

## 1. Importer les librairies et charger les données

Vous devez importer les librairies nécessaires.

Vous aurez notamment besoin de :

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
```

Chargez ensuite le fichier CSV fourni.

Vous devez afficher quelques informations de base sur le dataframe, par exemple :

```python
df.head()
df.info()
df.describe()
df.isna().sum()
```

Vous devez ajouter une courte phrase pour décrire ce que vous observez : nombre de colonnes, types de données, présence ou absence de valeurs manquantes.

---

## 2. Nettoyer et préparer les données

Vous devez préparer le dataset avant d’entraîner les modèles.

Votre préparation doit inclure les étapes suivantes :

1. convertir la colonne `Date` en format datetime ;
2. trier les données par ordre chronologique ;
3. vérifier les valeurs manquantes ;
4. créer la colonne `Target` ;
5. supprimer les lignes qui ne peuvent pas être utilisées pour l’entraînement.

Exemple attendu pour la colonne `Date` :

```python
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
```

Exemple attendu pour la cible :

```python
df["Target"] = df["Close"].shift(-1)
```

Après la création de `Target`, vous devez supprimer les lignes contenant des valeurs manquantes pour l’entraînement.

Attention : la colonne `Date` sert à comprendre l’ordre temporel des données, mais elle ne doit pas être utilisée directement comme variable explicative dans les modèles.

---

## 3. Produire deux visualisations pour comprendre les données

Pour que l’examen soit faisable en 2 heures, vous devez produire **deux visualisations obligatoires**.

### Graphique obligatoire 1 — Évolution du prix de clôture dans le temps

Vous devez représenter l’évolution de la colonne `Close` en fonction de la colonne `Date`.

Ce graphique doit permettre d’observer l’évolution générale du prix dans le temps.

Votre graphique doit contenir :

- un titre ;
- un nom pour l’axe des x ;
- un nom pour l’axe des y.

Après le graphique, ajoutez une courte interprétation.

---

### Graphique obligatoire 2 — Matrice de corrélation

Vous devez produire une matrice de corrélation avec les colonnes numériques suivantes :

```python
["Open", "High", "Low", "Close", "Volume", "Target"]
```

Cette visualisation doit vous aider à observer les liens entre les variables numériques et la variable cible.

Après le graphique, ajoutez une courte interprétation.

Dans votre interprétation, vous pouvez notamment expliquer quelles variables semblent fortement liées à `Target`.

---

## 4. Créer les variables explicatives `X` et la variable cible `y`

Vous devez créer les variables explicatives `X` avec les colonnes suivantes :

```python
features = ["Open", "High", "Low", "Close", "Volume"]
X = df[features]
```

Vous devez créer la variable cible `y` avec la colonne `Target` :

```python
y = df["Target"]
```

Attention : la colonne `Target` ne doit jamais être incluse dans `X`.

Si `Target` est incluse dans les variables explicatives, le modèle aura accès à la réponse pendant l’entraînement. Cela fausse les résultats.

---

## 5. Séparer les données d’entraînement et de test

Vous devez séparer les données en deux ensembles :

- un ensemble d’entraînement ;
- un ensemble de test.

Comme les données sont temporelles, les lignes ne doivent pas être mélangées au hasard.

Vous devez donc utiliser `shuffle=False`.

Code attendu :

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)
```

L’ensemble d’entraînement sert à entraîner les modèles. L’ensemble de test sert à évaluer les modèles sur des données qu’ils n’ont pas vues pendant l’entraînement.

---

## 6. Créer et entraîner les deux modèles

Vous devez entraîner deux modèles de régression.

Chaque modèle doit être placé dans un pipeline.

Un pipeline permet d’enchaîner proprement plusieurs étapes. Dans cet examen, chaque pipeline devra contenir :

1. un `RobustScaler` ;
2. un modèle de régression.

---

### Modèle 1 — Régression linéaire

Vous devez créer le pipeline suivant :

```python
linear_model = make_pipeline(
    RobustScaler(),
    LinearRegression()
)
```

Vous devez ensuite entraîner ce modèle sur les données d’entraînement.

---

### Modèle 2 — Lasso

Vous devez créer le pipeline suivant :

```python
lasso_model = make_pipeline(
    RobustScaler(),
    Lasso(alpha=0.001, max_iter=10000)
)
```

Vous devez ensuite entraîner ce modèle sur les données d’entraînement.

---

## 7. Calculer les scores R² des deux modèles

Après l’entraînement, vous devez faire des prédictions sur l’ensemble de test avec les deux modèles.

Vous devez ensuite calculer le score R² de chaque modèle avec `r2_score`.

Vous devez afficher clairement :

- le score R² du modèle `LinearRegression` ;
- le score R² du modèle `Lasso`.

Le meilleur modèle sera celui qui obtient le score R² le plus élevé sur les données de test.

---

## 8. Visualiser les prédictions des deux modèles

Vous devez créer **deux graphiques de prédiction**.

### Graphique 1 — Valeurs réelles et prédictions de LinearRegression

Ce graphique doit comparer :

- les valeurs réelles de `y_test` ;
- les valeurs prédites par le modèle `LinearRegression`.

---

### Graphique 2 — Valeurs réelles et prédictions de Lasso

Ce graphique doit comparer :

- les valeurs réelles de `y_test` ;
- les valeurs prédites par le modèle `Lasso`.

Chaque graphique doit contenir :

- un titre ;
- une légende ;
- un nom pour l’axe des x ;
- un nom pour l’axe des y.

Après les deux graphiques, ajoutez une courte interprétation.

Vous pouvez notamment expliquer lequel des deux modèles semble suivre le mieux les vraies valeurs.

---

## 9. Sélectionner le meilleur modèle

À partir des scores R² obtenus, vous devez déterminer quel modèle est le meilleur.

Vous devez afficher le nom du meilleur modèle.

Exemple de logique possible :

```python
if score_linear > score_lasso:
    best_model = linear_model
    best_model_name = "LinearRegression"
else:
    best_model = lasso_model
    best_model_name = "Lasso"

print("Meilleur modèle :", best_model_name)
```

Vous pouvez adapter cette structure à votre propre code.

---

## 10. Faire une prédiction avec le meilleur modèle

Une fois le meilleur modèle sélectionné, vous devez l’utiliser pour faire au moins une prédiction.

Pour rester dans un cadre vérifiable, vous devez faire une prédiction sur une ligne de l’ensemble de test.

Exemple :

```python
prediction = best_model.predict(X_test.iloc[[0]])
```

Vous devez afficher :

- le nom du meilleur modèle ;
- la valeur prédite ;
- la vraie valeur correspondante dans `y_test`.

Exemple :

```python
print("Meilleur modèle :", best_model_name)
print("Prédiction :", prediction[0])
print("Vraie valeur :", y_test.iloc[0])
```

Vous devez ajouter une courte phrase pour dire si la prédiction semble proche ou éloignée de la vraie valeur.

---

## 11. Question bonus — prédiction du lendemain de la dernière date du fichier

Cette question est optionnelle.

Vous pouvez utiliser le meilleur modèle pour prédire le prix de clôture du jour suivant la dernière date disponible dans le fichier.

Pour faire cela correctement, il faut conserver la dernière ligne du dataframe avant de supprimer les valeurs manquantes créées par `shift(-1)`.

Cette prédiction ne peut pas être comparée immédiatement à une vraie valeur, puisqu’elle correspond à une journée située après la dernière date disponible dans le dataset.

---

# Questions d’interprétation

À la fin du notebook, vous devez répondre brièvement aux questions suivantes.

1. Quel modèle obtient le meilleur score R² ?
2. Que signifie un score R² proche de 1 ?
3. Pourquoi la colonne `Target` ne doit-elle pas être incluse dans les variables explicatives `X` ?
4. Pourquoi utilise-t-on `shuffle=False` lors de la séparation des données ?
5. Le modèle peut-il prédire parfaitement le futur prix d’une cryptomonnaie ? Justifiez brièvement votre réponse.

---

# Contraintes techniques

Vous devez respecter les contraintes suivantes :

- utiliser uniquement les modèles demandés : `LinearRegression` et `Lasso` ;
- utiliser un pipeline pour chaque modèle ;
- utiliser `RobustScaler` dans chaque pipeline ;
- utiliser `r2_score` pour évaluer les modèles ;
- ne pas utiliser `SVR` ;
- ne pas utiliser `RandomForestRegressor` ;
- ne pas utiliser `PolynomialFeatures` ;
- ne pas inclure `Target` dans les variables explicatives ;
- ne pas mélanger les données temporelles au hasard lors de la séparation entraînement/test.

---

# Travail à remettre

Vous devez remettre un **Jupyter Notebook** au format `.ipynb`.

Le notebook doit être propre, lisible et bien commenté. Il doit contenir à la fois le code, les graphiques, les résultats obtenus et les explications écrites.

Vous devez remettre un notebook contenant :

1. le chargement des données ;
2. le nettoyage et la préparation du dataframe ;
3. les deux visualisations obligatoires ;
4. la création de `X` et `y` ;
5. la séparation des données ;
6. les deux pipelines ;
7. l’entraînement des deux modèles ;
8. les scores R² ;
9. les deux graphiques de prédiction ;
10. la sélection du meilleur modèle ;
11. une prédiction avec le meilleur modèle ;
12. les réponses aux questions d’interprétation.

Votre notebook doit être organisé, commenté et facile à suivre. Les cellules de code doivent être accompagnées de commentaires lorsque les étapes réalisées ne sont pas évidentes.

Les graphiques doivent avoir des titres et des axes clairement identifiés.

Les résultats doivent être accompagnés de courtes explications écrites dans des cellules Markdown. Il ne suffit pas d’afficher du code : vous devez montrer que vous comprenez les étapes réalisées.

---

# Remarque importante

Les cryptomonnaies sont des actifs très volatils. Même si un modèle obtient un bon score R² sur un ensemble de test, cela ne signifie pas qu’il peut prédire parfaitement le futur.

Un modèle de machine learning apprend à partir des données passées. Il peut repérer certaines tendances, mais il ne peut pas prévoir avec certitude les événements économiques, politiques, technologiques ou sociaux qui peuvent influencer fortement les marchés.

Dans cet examen, le but est donc d’évaluer votre capacité à construire et analyser un modèle de prédiction, et non de produire un outil financier fiable pour prendre des décisions d’investissement.
