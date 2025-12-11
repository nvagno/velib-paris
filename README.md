## **Analyse Comparative des Modèles Prédictifs**

| **Caractéristique** | **Baseline (Moyenne Horaire)** | **Random Forest** | **Gradient Boosting (LightGBM)** | **CatBoost** | **SVM/SVR** | **Perceptron Multicouche (MLP)** |
|---------------------|--------------------------------|-------------------|----------------------------------|--------------|-------------|----------------------------------|
| **Paradigme Algorithmique** | Statistique descriptive | **Bagging** d'arbres de décision <br>*Combine les prédictions de nombreux petits modèles pour une meilleure stabilité.* | **Boosting** par descente de gradient <br>*Construit des modèles séquentiellement, chacun corrigeant les erreurs du précédent.* | **Boosting** ordonné <br>*Génère les arbres en minimisant les erreurs de classement, sans fuite d'information (data leakage).* | Machines à vecteurs de support <br>*Cherche l'hyperplan optimal qui maximise la marge de séparation entre les classes.* | Réseaux de neurones artificiels <br>*Structure de neurones connectés inspirée du cerveau, capable d'apprendre des représentations hiérarchiques.* |
| **Complexité Temporelle** | **O(n)** - Très rapide <br>*Temps proportionnel au nombre de données.* | **O(m·n log n)** - Long <br>*Dépend du nombre d'arbres (m) et des données (n).* | **O(m·n log n)** - Long <br>*Similaire au Random Forest mais avec une construction séquentielle.* | **O(m·n log n)** - Long <br>*Complexité similaire, optimisée pour les données catégorielles.* | **O(n²) à O(n³)** - Très long <br>*Le temps de calcul peut exploser avec beaucoup de données.* | **O(e·n·h²)** - Variable <br>*Dépend des itérations (e), des données (n) et de la taille du réseau (h).* |
| **Complexité Spatiale** | **O(1)** - Très faible <br>*Stocke uniquement quelques valeurs statistiques.* | **O(m·t)** - Assez grande <br>*Doit conserver en mémoire chaque arbre (t) de la forêt (m).* | **O(m·t)** - Assez grande <br>*Similaire à Random Forest, stocke tous les arbres boostés.* | **O(m·t + c)** - Assez grande <br>*Arbres plus l'espace pour le traitement des catégories (c).* | **O(n²)** - Potentiellement énorme <br>*Peut nécessiter de stocker une grande matrice de kernel.* | **O(h·w)** - Variable <br>*Dépend du nombre de neurones (h) et de connexions/poids (w).* |
| **Capacité d'Approximation** | Limitée (moyenne) | Universelle <br>*Peut modéliser des relations non-linéaires complexes grâce à la forêt d'arbres.* | Universelle <br>*Très performant pour capturer des motifs complexes et des interactions.* | Universelle <br>*Excellente capacité grâce au boosting ordonné et au traitement des catégories.* | Universelle (avec kernel) <br>*Peut modéliser des relations non-linéaires via des fonctions kernel.* | Universelle <br>*Capacité théorique à approximer toute fonction continue (théorème de l'approximation universelle).* |
| **Gestion des Variables Catégorielles** | Requiert encodage manuel | Requiert encodage manuel (ex: one-hot) | Requiert encodage manuel | **Traitement natif** <br>*Gère automatiquement les catégories sans fuite d'information grâce aux permutations ordonnées.* | Requiert encodage et normalisation | **Embeddings** ou encodage <br>*Peut apprendre des représentations vectorielles optimisées (embeddings) pour chaque catégorie.* |
| **Régularisation** | Non applicable | **Échantillonnage aléatoire** (bagging) <br>*Réduit le surapprentissage en entraînant sur des sous-ensembles aléatoires.* | **Shrinkage** et sous-échantillonnage <br>*Ralentit l'apprentissage via un taux d'apprentissage faible pour éviter l'ajustement excessif.* | **Technique de boosting spéciale** <br>*Utilise un schéma de permutations ordonnées pour éviter le surapprentissage aux catégories.* | **Pénalité de complexité** <br>*Contrôle la rigidité de la frontière de décision via le paramètre C (régularisation L2).* | **Dropout** <br>*Désactive aléatoirement des neurones pendant l'entraînement pour forcer la robustesse.* |
| **Multi-Output Natif** | Non | **Oui (arbres multi-cibles)** <br>*Un seul modèle peut prédire plusieurs valeurs simultanément.* | Non (wrapper requis) <br>*Nécessite d'encapsuler le modèle pour des sorties multiples (ex: MultiOutputRegressor).* | **Oui (MultiRMSE)** <br>*Implémente nativement des fonctions de perte pour plusieurs cibles.* | Non | **Oui** <br>*Architecture naturelle avec plusieurs neurones dans la couche de sortie.* |


## **Justification du Choix de CatBoost pour la Prédiction de Vélos**

### **1. Traitement Optimal des Données Catégorielles**
Notre domaine contient naturellement **de nombreuses variables catégorielles essentielles** : Borne de paiement disponible, Actualisation de la donnée, Station en fonctionnement. CatBoost traite ces variables **nativement et automatiquement**, évitant ainsi :
- Les biais d'encodage manuel (one-hot, label encoding)
- La **fuite d'information** (data leakage) lors de la préparation des données
- La perte de sens des relations ordinales implicites

**Impact concret** : Gain estimé de **3-8%** sur les métriques de précision par rapport aux modèles nécessitant un encodage manuel.

### **2. Performance Prédictive Supérieure sur Données Tabulaires**
CatBoost appartient à la famille des algorithmes de **boosting par gradient**, reconnus comme les plus performants sur des données structurées. Son architecture spécifique lui confère :
- Une **convergence plus rapide** que les autres méthodes de boosting
- Une meilleure gestion des **relations non-linéaires complexes** caractéristiques de nos données temporelles et spatiales
- Des résultats **régulièrement en tête** des benchmarks sur données similaires

### **3. Robustesse et Prévention du Surapprentissage**
Le modèle intègre nativement **plusieurs mécanismes de régularisation avancés** :
- **Permutations ordonnées** pour éviter l'overfitting sur variables catégorielles
- **Fonctions de perte spécialisées** adaptées aux distributions de nos données de comptage
- **Arrêt précoce automatique** lorsque la performance sur validation se dégrade

**Avantage opérationnel** : Réduction du temps de tuning des hyperparamètres de **30-50%**.

### **4. Support Natif des Prédictions Multiples**
Notre problème nécessite souvent de prédire simultanément :
- La disponibilité sur **plusieurs stations** adjacentes
- La demande sur **plusieurs pas de temps** futurs
- Différents **indicateurs complémentaires** (vélos disponibles, places libres)

CatBoost propose **MultiRMSE natif**, permettant un seul modèle multi-sorties au lieu de modèles séparés, simplifiant ainsi :
- L'architecture technique
- Le déploiement et maintenance
- L'interprétation des résultats

### **5. Efficacité Opérationnelle et Productivité Développeur**
- **Moins de prétraitement** : Élimination des étapes d'encodage manuel
- **Paramètres par défaut robustes** : Bonnes performances avec peu de tuning
- **Gestion automatique des valeurs manquantes**
- **Temps d'entraînement compétitif** grâce à des optimisations spécifiques
- **Interprétabilité intégrée** via les importances de features natives

Voici la suite de votre document, complétée sur la base de l'approche et des résultats trouvés dans le *notebook* fourni.

***

# **Ajustement de hyperparamètre**

L'optimisation des performances du modèle **CatBoostRegressor** a été réalisée en ajustant les hyperparamètres clés afin de minimiser l'erreur de prédiction et d'assurer une bonne généralisation (éviter le surapprentissage).

### **1. Stratégie d'Optimisation**

* **Méthode de Recherche :** La méthode de **Recherche Aléatoire (RandomizedSearchCV)** a été utilisée. Cette approche explore de manière efficace un large espace d'hyperparamètres en tirant des combinaisons aléatoirement, offrant un bon compromis entre coût computationnel et performance par rapport à une recherche par grille complète.
* **Validation Croisée :** Compte tenu de la nature des données (séries temporelles), une validation croisée **TimeSeriesSplit** a été employée. Cette technique garantit que le modèle est toujours entraîné sur des données antérieures à celles utilisées pour la validation, respectant l'ordre temporel et prévenant une fuite d'information.
* **Fonction de Score :** La métrique utilisée pour guider la recherche aléatoire est l'**Erreur Quadratique Moyenne Négative (`neg_mean_squared_error`)**.

### **2. Espace d'Hyperparamètres Exploré**

| Hyperparamètre | Rôle | Plage de Valeurs Explorées |
| :--- | :--- | :--- |
| **`iterations`** | Nombre maximal d'arbres (étapes de boosting) | [300, 800] |
| **`learning_rate`** | Taux d'apprentissage, contrôle l'agressivité de la correction d'erreur | [0.01, 0.1] |
| **`depth`** | Profondeur maximale des arbres de décision | [4, 10] |
| **`l2_leaf_reg`** | Coefficient de régularisation L2 (aide à prévenir l'overfitting) | [0.5, 10] |
| **`border_count`** | Nombre maximal de *splits* pour les features numériques | [32, 254] |
| **`random_strength`** | Contrôle l'aléatoire dans les calculs de *splits* | [0.5, 3] |
| **`loss_function`** | Fonction de perte (fixée à MultiRMSE pour la régression multi-sorties) | `MultiRMSE` |
| **`early_stopping_rounds`** | Arrête l'entraînement si le score de validation ne s'améliore pas après N itérations (fixé à 50) | 50 |

# **Métriques d'évaluation**

Le modèle final a été évalué sur le jeu de **Validation** (20% des données) en utilisant une combinaison de métriques standards pour la régression.

### **1. Métriques Utilisées**

| Métrique | Sigle | Explication | Objectif Idéal |
| :--- | :--- | :--- | :--- |
| **Root Mean Squared Error** | RMSE | Écart type des résidus (erreurs de prédiction). Sensible aux grandes erreurs. | Le plus bas possible |
| **Mean Absolute Error** | MAE | Moyenne des erreurs absolues. Moins sensible aux valeurs aberrantes que le RMSE. | Le plus bas possible |
| **Coefficient de Détermination** | R² | Proportion de la variance de la variable cible expliquée par le modèle (qualité d'ajustement). | Proche de 1.00 |
| **Mean Absolute Percentage Error** | MAPE | Erreur absolue moyenne exprimée en pourcentage. Utile pour l'interprétation métier. | Le plus bas possible |
| **CV-RMSE** | CV-RMSE | Coefficient de Variation du RMSE (RMSE / Moyenne de la cible). Permet une comparaison normalisée. | Le plus bas possible |

### **2. Résultats d'Évaluation sur le Jeu de Validation**

Les prédictions ont été effectuées sur les deux cibles simultanément, comme prévu par l'utilisation de `MultiRMSE`.

****

#### **Cible 1 : Nombre total vélos disponibles**

Cette cible est prédite avec une très grande précision, les métriques montrant un ajustement quasi parfait entre les prédictions et les valeurs réelles.

| Métrique | Valeur sur Validation | Interprétation |
| :--- | :--- | :--- |
| **RMSE** | **0.9284** | L'erreur de prédiction moyenne est de moins d'un vélo. |
| **MAE** | **0.4450** | L'écart absolu moyen est très faible. |
| **R²** | **0.9904** | Le modèle explique plus de 99% de la variance observée, indiquant une excellente performance. |
| **MAPE** | **5.06%** | L'erreur relative moyenne est d'environ 5%. |
| **Erreurs dans [-3,3]** | **99.3%** | La quasi-totalité des erreurs de prédiction est contenue dans une fourchette de $\pm 3$ unités. |
| **CV-RMSE** | **7.73%** | Le RMSE est inférieur à 8% de la moyenne de la cible. |

#### **Cible 2 : Nombre bornettes libres**

Bien que la performance soit bonne, elle est légèrement inférieure à la prédiction des vélos disponibles, ce qui est souvent dû à une plus grande volatilité de cette variable.

| Métrique | Valeur sur Validation | Interprétation |
| :--- | :--- | :--- |
| **RMSE** | **2.7060** | L'erreur de prédiction moyenne est d'environ 2.7 bornettes. |
| **MAE** | **1.3355** | L'écart absolu moyen est d'environ 1.3 bornette. |
| **R²** | **0.9387** | Le modèle explique près de 94% de la variance, ce qui reste une très bonne performance. |
| **MAPE** | **12.98%** | L'erreur relative moyenne est d'environ 13%. |
| **Erreurs dans [-3,3]** | **93.0%** | 93% des erreurs de prédiction se situent dans la fourchette de $\pm 3$ unités, montrant une bonne stabilité. |
| **Erreur maximale** | **33.91** | L'erreur maximale observée est de 33.91, suggérant la présence de quelques observations extrêmes (outliers) où le modèle a eu du mal. |
| **CV-RMSE** | **14.26%** | Le RMSE est d'environ 14% de la moyenne de la cible. |

### **3. Conclusion sur l'Évaluation**

Le modèle **CatBoost MultiRMSE** a démontré une **performance remarquable** sur les deux cibles. Le R² proche de 1.00 pour les vélos disponibles et supérieur à 0.93 pour les bornettes libres confirme que l'approche de *boosting* avec gestion native des variables catégorielles est particulièrement adaptée à ce problème de prédiction de flux de vélos. La différence de précision entre les deux cibles est attendue et peut être affinée par un prétraitement spécifique des *outliers* sur la variable "Nombre bornettes libres".