import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Simulare date
np.random.seed(42)
n_users = 100
n_modules = 5
data = []
for user_id in range(1, n_users + 1):
    for module_id in range(1, n_modules + 1):
        video_length = np.random.uniform(3, 7)
        time_spent = np.random.uniform(0.5, 1.2) * video_length
        time_spent = min(time_spent, video_length)
        quiz_score = np.random.normal(loc=75 + (time_spent / video_length) * 25, scale=10)
        quiz_score = max(0, min(quiz_score, 100))
        completed = 1 if time_spent >= 0.9 * video_length else 0
        data.append([user_id, module_id, video_length, time_spent, quiz_score, completed])

df = pd.DataFrame(data, columns=["user_id", "module_id", "video_length", "time_spent", "quiz_score", "completed"])

st.title("📊 Analiză Microlearning pe Bază de Clickstream")
st.write("Această aplicație simulează și analizează date clickstream pentru a evalua impactul microlearning.")

# Selectarea unui modul pentru filtrare
selected_module = st.selectbox("Selectează un modul:", sorted(df["module_id"].unique()))
df_filtered = df[df["module_id"] == selected_module]

# Vizualizare heatmap
st.subheader("📘 Scoruri medii în funcție de finalizare")
heatmap_data = df.pivot_table(index="module_id", columns="completed", values="quiz_score", aggfunc="mean")
fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax1)
st.pyplot(fig1)

# Regresie
X = df_filtered[["time_spent", "video_length"]]
y = df_filtered["quiz_score"]
reg_model = LinearRegression().fit(X, y)
r_squared = reg_model.score(X, y)

st.subheader("📈 Model de regresie")
st.write("**R²:**", round(r_squared, 3))
st.write("**Coeficienți:**", dict(zip(X.columns, reg_model.coef_)))

# Clustering
scaler = StandardScaler()
df_clustered = df_filtered.copy()
df_clustered[["time_spent", "quiz_score"]] = scaler.fit_transform(df_clustered[["time_spent", "quiz_score"]])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clustered["cluster"] = kmeans.fit_predict(df_clustered[["time_spent", "quiz_score"]])

# Vizualizare clustering
st.subheader("🧠 Clustering comportamental")
fig2, ax2 = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=df_clustered, x="time_spent", y="quiz_score", hue="cluster", palette="Set2", ax=ax2)
plt.xlabel("Timp petrecut (standardizat)")
plt.ylabel("Scor la quiz (standardizat)")
st.pyplot(fig2)

# Date brute
if st.checkbox("🔍 Afișează datele brute"):
    st.dataframe(df_filtered.reset_index(drop=True))
