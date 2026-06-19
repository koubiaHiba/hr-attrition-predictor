# app.py - Version complète pour votre projet Data Mining
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (homogeneity_score, v_measure_score,
                              adjusted_rand_score, calinski_harabasz_score)
from mlxtend.frequent_patterns import apriori, association_rules

# Configuration de la page
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="🔮",
    layout="wide"
)

# Titre centré
st.markdown("""
    <div style="text-align: center;">
        <h1>📊 Data Mining Project</h1>
        <h2 style="color: #2ecc71;">HR Attrition Predictor</h2>
        <hr style="width: 50%; margin: auto;">
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Chargement des modèles
@st.cache_resource
def load_models():
    models = {}
    try:
        models['classifier'] = joblib.load('best_model.pkl')
        models['scaler'] = joblib.load('scaler.pkl')
        models['features'] = joblib.load('feature_names.pkl')
        st.success("✅ Modèles chargés avec succès")
    except Exception as e:
        st.error(f"⚠️ Erreur de chargement: {e}")
        st.info("Assurez-vous que les fichiers 'best_model.pkl', 'scaler.pkl' et 'feature_names.pkl' sont présents")
    return models

models = load_models()

# Clustering réel (K-Means) — remplace l'ancien groupby décoratif par JobSatisfaction.
# Calculé une seule fois par session grâce à @st.cache_data. SSE choisit K (slide K-means) ;
# Homogeneity/V-measure/ARI/Calinski-Harabasz valident les clusters contre Attrition (slide
# Validation) — deux rôles distincts, pas deux façons concurrentes de mesurer la même chose.
@st.cache_data
def run_clustering():
    df = pd.read_csv('dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    zero_var_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df2 = df.drop(columns=zero_var_drop)
    y_attrition = (df2['Attrition'] == 'Yes').astype(int).values
    X_raw = df2.drop(columns=['Attrition'])
    cat_cols = X_raw.select_dtypes(include='object').columns.tolist()
    X_enc = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    X_enc = X_enc[models['features']]
    X_scaled = models['scaler'].transform(X_enc)

    # SSE par K (méthode du coude) — le cours nomme explicitement SSE comme "Mesure
    # d'évaluation" pour l'algorithme K-means lui-même (slide Algorithmes : K-means).
    k_range = list(range(2, 9))
    sse_by_k = []
    for k in k_range:
        km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_k.fit(X_scaled)
        sse_by_k.append(km_k.inertia_)

    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = km.fit_predict(X_scaled)

    # Métriques de VALIDATION nommées par le cours pour le Clustering (slide "Fonction coût :
    # Mesures d'évaluation") : compare les clusters à la variable "classe" masquée (Attrition),
    # stratégie explicitement décrite par le cours comme "la plus utilisée" (slide Validation).
    # Différent du SSE ci-dessus, qui sert à choisir K, pas à valider contre Attrition.
    metrics = {
        'Homogeneity': homogeneity_score(y_attrition, clusters),
        'V-measure': v_measure_score(y_attrition, clusters),
        'Adjusted Rand Index': adjusted_rand_score(y_attrition, clusters),
        'Calinski-Harabasz': calinski_harabasz_score(X_scaled, clusters),
    }

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_.sum()

    profile = df[['Age', 'MonthlyIncome', 'YearsAtCompany']].copy()
    profile['Cluster'] = clusters
    profile['Attrition (%)'] = y_attrition * 100
    profile_summary = profile.groupby('Cluster').mean().round(2)
    profile_summary.columns = ['Âge moyen', 'Revenu moyen', 'Ancienneté moyenne', 'Attrition (%)']

    return clusters, X_pca, var_explained, metrics, profile_summary, y_attrition, k_range, sse_by_k


# Règles d'association (Apriori) — Technique Descriptive nommée par le cours,
# absente de la version précédente de l'application.
@st.cache_data
def run_association_rules():
    df = pd.read_csv('dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    df_t = pd.DataFrame(index=df.index)
    df_t['Age'] = pd.cut(df['Age'], bins=[17, 30, 40, 50, 61],
                          labels=['Age_<30', 'Age_30-40', 'Age_40-50', 'Age_50+'])
    df_t['Income'] = pd.qcut(df['MonthlyIncome'], q=3,
                              labels=['Income_Bas', 'Income_Moyen', 'Income_Haut'])
    df_t['Tenure'] = pd.cut(df['YearsAtCompany'], bins=[-1, 2, 5, 10, 100],
                             labels=['Tenure_0-2ans', 'Tenure_3-5ans', 'Tenure_6-10ans', 'Tenure_10ans+'])
    for col, label in [('JobSatisfaction', 'JobSat'), ('WorkLifeBalance', 'WLB')]:
        df_t[col] = label + '_' + df[col].astype(str)
    for col in ['OverTime', 'MaritalStatus', 'JobRole']:
        df_t[col] = col + '_' + df[col].astype(str).str.replace(' ', '_')
    df_t['Attrition'] = 'Attrition_' + df['Attrition']

    # prefix="" : les valeurs ci-dessus sont déjà auto-descriptives. Le préfixe par défaut
    # de pandas double le nom (ex: "Attrition_Attrition_Yes") et casse le filtrage en aval.
    basket = pd.get_dummies(df_t, prefix="", prefix_sep="")
    frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True, max_len=3)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    rules_yes = rules[rules['consequents'].apply(
        lambda x: 'Attrition_Yes' in x and len(x) == 1)].sort_values('lift', ascending=False).copy()
    rules_yes['Antécédent'] = rules_yes['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    base_rate = (df['Attrition'] == 'Yes').mean()

    return rules_yes[['Antécédent', 'support', 'confidence', 'lift']].rename(
        columns={'support': 'Support', 'confidence': 'Confiance', 'lift': 'Lift'}), base_rate

# Sidebar de navigation (ORDRE MODIFIÉ)
option = st.sidebar.selectbox(
    "📌 Choisissez une analyse",
    ["📈 Visualisation", "🔮 Prédiction", "📊 Clustering", "🔗 Association", "📄 Rapport"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### À propos")
st.sidebar.info(
    "Cette application prédit le risque d'attrition des employés "
    "en utilisant un modèle de Régression Logistique entraîné sur "
    "le dataset IBM HR Analytics."
)

# ==================== SECTION 1 : VISUALISATION ====================
if option == "📈 Visualisation":
    st.header("📈 Visualisation des données")
    
    try:
        df = pd.read_csv('dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
        
        st.subheader("📋 Aperçu des données")
        st.dataframe(df.head(10))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribution de l'attrition")
            attrition_counts = df['Attrition'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(attrition_counts, labels=attrition_counts.index, autopct='%1.1f%%', 
                   colors=['#2ecc71', '#e74c3c'], explode=(0.05, 0))
            ax.set_title("Répartition des employés (Attrition)")
            st.pyplot(fig)
            
            st.caption(f"Total employés : {len(df)} | Départs : {attrition_counts.get('Yes', 0)}")
        
        with col2:
            st.subheader("⏰ Impact des heures supplémentaires")
            overtime_attrition = pd.crosstab(df['OverTime'], df['Attrition'], normalize='index') * 100
            fig, ax = plt.subplots()
            overtime_attrition.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
            ax.set_ylabel('Pourcentage (%)')
            ax.set_title("Taux d'attrition selon les heures supplémentaires")
            ax.legend(title='Attrition')
            ax.set_xticklabels(['Non', 'Oui'], rotation=0)
            st.pyplot(fig)
        
        st.subheader("🔥 Matrice de corrélation")
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:12]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', 
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title("Corrélation entre les variables numériques")
        st.pyplot(fig)
        
        st.subheader("📈 Distribution des variables clés")
        col3, col4 = st.columns(2)
        
        with col3:
            fig, ax = plt.subplots()
            df['Age'].hist(bins=20, ax=ax, color='skyblue', edgecolor='black')
            ax.set_xlabel('Âge')
            ax.set_ylabel('Nombre d\'employés')
            ax.set_title('Distribution des âges')
            st.pyplot(fig)
        
        with col4:
            fig, ax = plt.subplots()
            df['MonthlyIncome'].hist(bins=20, ax=ax, color='lightgreen', edgecolor='black')
            ax.set_xlabel('Revenu mensuel (DH)')
            ax.set_ylabel('Nombre d\'employés')
            ax.set_title('Distribution des revenus')
            st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Erreur de chargement du dataset: {e}")

# ==================== SECTION 2 : PRÉDICTION ====================
elif option == "🔮 Prédiction":
    st.header("🔮 Prédiction du risque d'attrition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Âge", 18, 60, 36)
        monthly_income = st.number_input("Revenu mensuel (DH)", 1000, 20000, 6500)
        years_at_company = st.slider("Années dans l'entreprise", 0, 40, 5)
        job_satisfaction = st.slider("Satisfaction au travail", 1, 4, 3)
        distance_from_home = st.slider("Distance domicile-travail (km)", 1, 29, 7)
    
    with col2:
        overtime = st.selectbox("Heures supplémentaires", ["No", "Yes"])
        environment_satisfaction = st.slider("Satisfaction environnement", 1, 4, 3)
        work_life_balance = st.slider("Équilibre vie pro/perso", 1, 4, 3)
        years_since_last_promotion = st.slider("Années depuis dernière promotion", 0, 15, 2)
        job_involvement = st.slider("Implication au travail", 1, 4, 3)
    
    if st.button("🔍 Prédire le risque", type="primary"):
        if models:
            # Construction des données
            input_dict = {
                'Age': age,
                'DailyRate': 800,
                'DistanceFromHome': distance_from_home,
                'Education': 3,
                'EnvironmentSatisfaction': environment_satisfaction,
                'HourlyRate': 66,
                'JobInvolvement': job_involvement,
                'JobLevel': 2,
                'JobSatisfaction': job_satisfaction,
                'MonthlyIncome': monthly_income,
                'MonthlyRate': 15000,
                'NumCompaniesWorked': 2,
                'PercentSalaryHike': 15,
                'PerformanceRating': 3,
                'RelationshipSatisfaction': 3,
                'StockOptionLevel': 1,
                'TotalWorkingYears': years_at_company + 5,
                'TrainingTimesLastYear': 2,
                'WorkLifeBalance': work_life_balance,
                'YearsAtCompany': years_at_company,
                'YearsInCurrentRole': max(0, years_at_company - 1),
                'YearsSinceLastPromotion': years_since_last_promotion,
                'YearsWithCurrManager': max(0, years_at_company - 1),
                'BusinessTravel_Travel_Rarely': 1,
                'Department_Research & Development': 1,
                'EducationField_Life Sciences': 1,
                'Gender_Male': 1,
                'JobRole_Sales Executive': 1,
                'MaritalStatus_Married': 1,
                'OverTime_Yes': 1 if overtime == "Yes" else 0,
            }
            
            # Création du DataFrame
            input_df = pd.DataFrame([input_dict])
            input_df = input_df.reindex(columns=models['features'], fill_value=0)
            
            # Scaling et prédiction
            input_scaled = models['scaler'].transform(input_df)
            prediction = models['classifier'].predict(input_scaled)[0]
            probability = models['classifier'].predict_proba(input_scaled)[0][1]
            
            st.markdown("---")
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediction == 1:
                    st.error("⚠️ **RISQUE ÉLEVÉ d'attrition**")
                else:
                    st.success("✅ **RISQUE FAIBLE d'attrition**")
            
            with col_res2:
                st.metric("📊 Probabilité", f"{probability:.1%}")
            
            if prediction == 1:
                st.warning("""
                **📋 Recommandations :**
                - ✨ Réduire les heures supplémentaires
                - 🎯 Améliorer la satisfaction au travail
                - 📈 Envisager une promotion ou une augmentation
                - 💼 Améliorer l'équilibre vie pro/perso
                """)

# ==================== SECTION 3 : CLUSTERING ====================
elif option == "📊 Clustering":
    st.header("📊 Analyse de Clustering (K-Means)")
    st.markdown("Segmentation non supervisée des employés (K=2), validée a posteriori contre l'Attrition réelle")

    try:
        clusters, X_pca, var_explained, metrics, profile_summary, y_attrition, k_range, sse_by_k = run_clustering()

        st.subheader("📉 Choix de K : méthode du coude (SSE)")
        st.caption("Le cours nomme SSE (Sum of Squared Errors) comme mesure d'évaluation de K-means "
                   "lui-même — sert à choisir K, indépendamment de toute comparaison avec Attrition.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k_range, sse_by_k, marker='o', color='#2ecc71')
        ax.axvline(x=2, color='#e74c3c', linestyle='--', alpha=0.6, label='K=2 retenu')
        ax.set_xlabel('K (nombre de clusters)')
        ax.set_ylabel('SSE (inertie)')
        ax.set_title('Évolution du SSE selon K')
        ax.legend()
        st.pyplot(fig)
        st.caption("La décroissance est progressive plutôt qu'un coude net — attendu sur des données "
                   "RH à 44 dimensions après encodage. Le choix final de K=2 s'appuie surtout sur le "
                   "pic du Calinski-Harabasz Index ci-dessous, le SSE servant de diagnostic complémentaire.")

        st.subheader("📐 Métriques de validation (clusters vs Attrition réelle)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Homogeneity", f"{metrics['Homogeneity']:.4f}")
        m2.metric("V-measure", f"{metrics['V-measure']:.4f}")
        m3.metric("Adjusted Rand Index", f"{metrics['Adjusted Rand Index']:.4f}")
        m4.metric("Calinski-Harabasz", f"{metrics['Calinski-Harabasz']:.1f}")
        st.caption(
            "Homogeneity / V-measure / ARI proches de 0 : les clusters géométriques ne recoupent pas "
            "exactement les groupes d'Attrition réels — attendu, pas une erreur (sinon la classification "
            "supervisée serait quasi parfaite et non à F1≈0.5). Calinski-Harabasz est maximal à K=2, ce qui "
            "justifie ce choix indépendamment du fait qu'Attrition soit elle-même binaire."
        )

        st.subheader("🎯 Projection PCA et composition des clusters")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(7, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=40)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(f'K-Means (K=2) — projection PCA 2D ({var_explained*100:.0f}% variance)')
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(fig)

        with col2:
            ct = pd.crosstab(clusters, y_attrition, normalize='index') * 100
            ct.columns = ['Non', 'Oui']
            ct.index.name = 'Cluster'
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(ct, annot=True, fmt='.1f', cmap='Blues', ax=ax, cbar_kws={'label': '% du cluster'})
            ax.set_title('Composition de chaque cluster (% Attrition réelle)')
            st.pyplot(fig)

        st.caption(f"⚠️ La projection PCA ne capture que {var_explained*100:.0f}% de la variance totale — "
                   "simplification visuelle pour l'affichage, pas la structure complète à 44 dimensions.")

        st.subheader("📊 Profil des clusters")
        st.dataframe(profile_summary, use_container_width=True)

    except Exception as e:
        st.warning(f"Erreur de chargement: {e}")

# ==================== SECTION : ASSOCIATION ====================
elif option == "🔗 Association":
    st.header("🔗 Règles d'association (Apriori)")
    st.markdown("Quelles combinaisons de profils sont associées à un risque d'attrition supérieur à la moyenne ?")

    try:
        rules_yes, base_rate = run_association_rules()

        st.caption(f"Taux de base Attrition='Yes' dans la population : {base_rate*100:.1f}% — "
                   "une règle n'est intéressante que si son Lift dépasse nettement 1.0.")

        if len(rules_yes) == 0:
            st.info("Aucune règle trouvée avec ces seuils de support/lift.")
        else:
            st.dataframe(
                rules_yes.style.format({'Support': '{:.3f}', 'Confiance': '{:.3f}', 'Lift': '{:.2f}x'}),
                use_container_width=True
            )
            st.caption("Support et Confiance sont les métriques nommées par le cours pour l'Association ; "
                       "le Lift est ajouté pour distinguer une vraie découverte d'un simple effet de taux de base "
                       "(à 16% d'Attrition='Yes', presque toute règle a une confiance élevée vers 'Non').")

    except Exception as e:
        st.warning(f"Erreur de chargement: {e}")

# ==================== SECTION 4 : RAPPORT ====================
elif option == "📄 Rapport":
    st.header("📄 Rapport du projet")
    
    st.markdown("""
    ---
    ### 🎯 Objectif du projet
    Prédire le risque d'attrition des employés (départ volontaire de l'entreprise)
    afin de permettre à la direction de mettre en place des actions de rétention.
    
    ---
    ### 📊 Méthodologie (CRISP-DM)
    
    | Phase | Description |
    |-------|-------------|
    | **1. Business Understanding** | Compréhension du problème et des objectifs |
    | **2. Data Understanding** | Analyse exploratoire du dataset (1470 employés, 35 variables) |
    | **3. Data Preparation** | Nettoyage, encodage, SMOTE, normalisation |
    | **4. Modeling** | Test de 6 algorithmes de classification |
    | **5. Evaluation** | Comparaison avec métriques (Accuracy, MCC) |
    | **6. Deployment** | Application Streamlit interactive |
    
    ---
    ### 🏆 Résultats des modèles
    
    | Modèle | Accuracy | Précision | Rappel | F1-score | MCC |
    |--------|----------|-----------|--------|----------|-----|
    | Régression Logistique | 85.03% | 0.541 | 0.426 | 0.476 | **0.394** |
    | SVM | 85.03% | 0.548 | 0.362 | 0.436 | 0.364 |
    | Random Forest | 82.31% | 0.400 | 0.213 | 0.278 | 0.200 |
    | Arbre de décision | 71.77% | 0.263 | 0.426 | 0.325 | 0.166 |
    | KNN | 67.69% | 0.239 | 0.468 | 0.317 | 0.146 |
    | Naive Bayes | 60.20% | 0.208 | 0.532 | 0.299 | 0.110 |
    
    **🏆 Meilleur modèle : Régression Logistique**
    
    ---
    ### 🔑 Facteurs clés identifiés
    - ⚠️ **Heures supplémentaires** : Augmentent fortement le risque
    - ✅ **Satisfaction au travail** : Facteur protecteur
    - ✅ **Ancienneté** : Réduit le risque
    - ⚠️ **Équilibre vie pro/perso** : Important à maintenir
    
    ---
    ### 🔬 Analyses complémentaires (ajoutées)
    En plus du pipeline original, 4 analyses ont été ajoutées :
    - **Validation croisée (StratifiedKFold-10) + GridSearchCV** : confirme l'absence d'overfitting
      pour la Régression Logistique (écart Train/Test ≈ 0pp, contre 13-20pp pour SVM/RF/Arbre/KNN)
    - **Courbe ROC-AUC** : AUC = 0.80 pour le modèle final
    - **Clustering K-Means (K=2)**, ci-contre dans l'onglet 📊 — remplace l'ancien groupby par
      JobSatisfaction, évalué avec les métriques nommées par le cours (Homogeneity, V-measure,
      Adjusted Rand Index, Calinski-Harabasz)
    - **Règles d'association (Apriori)**, onglet 🔗 — ex: Revenu bas + ancienneté 0-2 ans → Attrition
      (Lift 2.27x)

    ---
    ### 📁 Fichiers du projet
    - `principal.ipynb` : Notebook complet (Phases 1-4)
    - `best_model.pkl` : Modèle sauvegardé
    - `scaler.pkl` : Scaler sauvegardé
    - `feature_names.pkl` : Noms des colonnes
    - `app.py` : Application Streamlit
    - `dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv` : Données
    
    ---
    ### 👥 Équipe
    Projet réalisé dans le cadre du module **Data Mining**
    - **Hiba Koubia**
    - **Youssef Nita**            
    """)

st.sidebar.markdown("---")
st.sidebar.caption("🔮 Projet Data Mining - SDSI | Promotion 2026")