import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# ── CONFIGURAÇÕES ─────────────────────────────────────────────────
PATH = r"C:\Users\emill\.cache\kagglehub\datasets\gustavoubeda\complaints-from-reclame-aqui-and-consumidor-gov\versions\2\reclamacoes.csv"
AMOSTRA = 100000

print("=" * 60)
print(" ANÁLISE DE RECLAMAÇÕES — RECLAME AQUI")
print("=" * 60)

# ── 1. CARREGAMENTO ───────────────────────────────────────────────
print("\n[1/7] Carregando dados...")
df = pd.read_csv(PATH, nrows=AMOSTRA, low_memory=False)
df = df[["ask", "company", "resolved", "uf_ask", "class_note"]].dropna(subset=["ask"])
df["resolved_bin"] = df["resolved"].str.lower().str.contains("não").astype(int)
print(f"Registros carregados: {len(df)}")

# ── 2. EDA ────────────────────────────────────────────────────────
print("\n[2/7] Análise exploratória...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Empresas mais reclamadas
top_empresas = df["company"].value_counts().head(10)
top_empresas.plot.barh(ax=axes[0], color="#D85A30", edgecolor="white")
axes[0].set_title("Top 10 empresas mais reclamadas")
axes[0].set_xlabel("Número de reclamações")
axes[0].invert_yaxis()

# Resolvido vs não resolvido
df["resolved"].value_counts().plot.bar(
    ax=axes[1], color=["#1D9E75", "#D85A30", "#E8A020"], edgecolor="white"
)
axes[1].set_title("Status das reclamações")
axes[1].set_xlabel("")
axes[1].tick_params(rotation=30)

# Reclamações por estado
top_uf = df["uf_ask"].value_counts().head(10)
top_uf.plot.bar(ax=axes[2], color="#378ADD", edgecolor="white")
axes[2].set_title("Top 10 estados com mais reclamações")
axes[2].set_xlabel("Estado")
axes[2].tick_params(rotation=0)

plt.tight_layout()
plt.savefig("eda_reclamacoes.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico EDA salvo.")

# ── 3. PRÉ-PROCESSAMENTO DE TEXTO ─────────────────────────────────
print("\n[3/7] Pré-processamento de texto...")

stop_pt = set(stopwords.words("portuguese"))
stop_pt.update(["empresa", "produto", "serviço", "dia", "dias", "vez", "fazer"])

def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-záàâãéèêíïóôõúücç\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    palavras = [p for p in texto.split() if p not in stop_pt and len(p) > 2]
    return " ".join(palavras)

df["texto_limpo"] = df["ask"].apply(limpar_texto)
print(f"Exemplo original: {df['ask'].iloc[0][:100]}...")
print(f"Exemplo limpo:    {df['texto_limpo'].iloc[0][:100]}...")

# ── 4. NUVEM DE PALAVRAS ──────────────────────────────────────────
print("\n[4/7] Gerando nuvens de palavras...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, (label, titulo, cor) in zip(axes, [
    (0, "Reclamações RESOLVIDAS", "#1D9E75"),
    (1, "Reclamações NÃO RESOLVIDAS", "#D85A30")
]):
    texto = " ".join(df[df["resolved_bin"] == label]["texto_limpo"].dropna())
    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap="Greens" if label == 0 else "Reds",
        max_words=100
    ).generate(texto)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig("wordcloud_reclamacoes.png", dpi=150, bbox_inches="tight")
plt.show()
print("Nuvens de palavras salvas.")

# ── 5. PALAVRAS MAIS FREQUENTES ───────────────────────────────────
print("\n[5/7] Analisando palavras mais frequentes por empresa...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
top3_empresas = df["company"].value_counts().head(6).index

for ax, empresa in zip(axes.flatten(), top3_empresas):
    textos = " ".join(df[df["company"] == empresa]["texto_limpo"].dropna())
    palavras = Counter(textos.split()).most_common(10)
    words, counts = zip(*palavras)
    ax.barh(words, counts, color="#534AB7", edgecolor="white")
    ax.set_title(f"{empresa}", fontsize=12, fontweight="bold")
    ax.invert_yaxis()

plt.suptitle("Palavras mais frequentes por empresa", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("palavras_por_empresa.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico de palavras salvo.")

# ── 6. MODELO DE CLASSIFICAÇÃO ────────────────────────────────────
print("\n[6/7] Treinando modelo de classificação...")

df_model = df[["texto_limpo", "resolved_bin"]].dropna()
df_model = df_model[df_model["texto_limpo"].str.len() > 10]

X_train, X_test, y_train, y_test = train_test_split(
    df_model["texto_limpo"], df_model["resolved_bin"],
    test_size=0.2, random_state=42, stratify=df_model["resolved_bin"]
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

print("\n── RELATÓRIO DE CLASSIFICAÇÃO ──")
print(classification_report(y_test, y_pred,
      target_names=["Resolvido", "Não Resolvido"]))

fig, ax = plt.subplots(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Resolvido", "Não Resolvido"],
            yticklabels=["Resolvido", "Não Resolvido"], ax=ax)
ax.set_title("Matriz de Confusão — Classificação de Reclamações")
ax.set_ylabel("Real")
ax.set_xlabel("Previsto")
plt.tight_layout()
plt.savefig("modelo_classificacao.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 7. TAXA DE RESOLUÇÃO POR EMPRESA ─────────────────────────────
print("\n[7/7] Calculando taxa de resolução por empresa...")

taxa = df.groupby("company").agg(
    total=("resolved_bin", "count"),
    nao_resolvidos=("resolved_bin", "sum")
).reset_index()
taxa["taxa_resolucao"] = (1 - taxa["nao_resolvidos"] / taxa["total"]) * 100
taxa = taxa[taxa["total"] > 100].sort_values("taxa_resolucao", ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))
cores = ["#D85A30" if v < 70 else "#1D9E75" for v in taxa["taxa_resolucao"]]
ax.barh(taxa["company"], taxa["taxa_resolucao"], color=cores, edgecolor="white")
ax.axvline(x=70, color="gray", linestyle="--", alpha=0.5, label="Meta 70%")
ax.set_title("Taxa de resolução por empresa — Reclame Aqui", fontsize=14, fontweight="bold")
ax.set_xlabel("% de reclamações resolvidas")
ax.legend()
plt.tight_layout()
plt.savefig("taxa_resolucao_empresa.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ Análise completa! Arquivos gerados:")
print("  - eda_reclamacoes.png")
print("  - wordcloud_reclamacoes.png")
print("  - palavras_por_empresa.png")
print("  - modelo_classificacao.png")
print("  - taxa_resolucao_empresa.png")