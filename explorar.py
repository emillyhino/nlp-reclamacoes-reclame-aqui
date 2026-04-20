import pandas as pd

path = r"C:\Users\emill\.cache\kagglehub\datasets\gustavoubeda\complaints-from-reclame-aqui-and-consumidor-gov\versions\2\reclamacoes.csv"

df = pd.read_csv(path)
print(f"Shape: {df.shape}")
print(f"\nEmpresas únicas: {df['company'].nunique()}")
print(f"Estados únicos: {df['uf_ask'].nunique()}")
print(f"Resolvidos: {df['resolved'].value_counts().to_dict()}")
print(f"\nNotas únicas: {df['class_note'].unique()}")
print(f"\nExemplo de reclamação:")
print(df['ask'].iloc[0])
print(f"\nTop 10 empresas mais reclamadas:")
print(df['company'].value_counts().head(10))