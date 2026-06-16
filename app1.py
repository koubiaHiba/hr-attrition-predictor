# app.py - Version complète pour votre projet Data Mining
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
        models['classifier'] = joblib.load('models/best_model.pkl')
        models['scaler'] = joblib.load('models/scaler.pkl')
        models['features'] = joblib.load('models/feature_names.pkl')
        st.success("✅ Modèles chargés avec succès")
    except Exception as e:
        st.error(f"⚠️ Erreur de chargement: {e}")
        st.info("Assurez-vous que les fichiers 'best_model.pkl', 'scaler.pkl' et 'feature_names.pkl' sont présents")
    return models

models = load_models()

# Sidebar de navigation (ORDRE MODIFIÉ)
option = st.sidebar.selectbox(
    "📌 Choisissez une analyse",
    ["📈 Visualisation", "🔮 Prédiction", "📊 Clustering", "📄 Rapport"]
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
    st.header("📊 Analyse de Clustering")
    st.markdown("Segmentation des employés basée sur leurs caractéristiques")
    
    try:
        df = pd.read_csv('dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
        
        st.subheader("🎯 Visualisation des segments d'employés")
        
        # Sélection des features pour le clustering
        cluster_features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(df['Age'], df['MonthlyIncome'], 
                               c=df['JobSatisfaction'], cmap='viridis', alpha=0.6, s=50)
            ax.set_xlabel('Âge')
            ax.set_ylabel('Revenu mensuel (DH)')
            ax.set_title('Segmentation Âge vs Revenu')
            plt.colorbar(scatter, label='Satisfaction au travail')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(df['YearsAtCompany'], df['MonthlyIncome'], 
                               c=df['WorkLifeBalance'], cmap='plasma', alpha=0.6, s=50)
            ax.set_xlabel('Années dans l\'entreprise')
            ax.set_ylabel('Revenu mensuel (DH)')
            ax.set_title('Segmentation Ancienneté vs Revenu')
            plt.colorbar(scatter, label='Équilibre vie pro/perso')
            st.pyplot(fig)
        
        st.info("📌 Les couleurs représentent respectivement la satisfaction au travail et l'équilibre vie pro/perso")
        
        # Statistiques par cluster
        st.subheader("📊 Profil des employés par satisfaction")
        satisfaction_groups = df.groupby('JobSatisfaction').agg({
            'Age': 'mean',
            'MonthlyIncome': 'mean',
            'YearsAtCompany': 'mean'
        }).round(2)
        satisfaction_groups.columns = ['Âge moyen', 'Revenu moyen', 'Ancienneté moyenne']
        st.dataframe(satisfaction_groups)
        
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