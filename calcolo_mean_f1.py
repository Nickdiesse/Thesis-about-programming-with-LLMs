import psycopg2

# Funzione per connettersi al database e recuperare i F1 score
def retrieve_f1_scores():
    conn = psycopg2.connect(
        host="localhost",
        database="benchmarkllm",
        user="postgres",
        password="nicola"
    )
    cursor = conn.cursor()

    # Recupera i F1 score per ciascun modello
    cursor.execute("SELECT f1_llama, f1_orca, f1_falcon FROM benchmark_results")
    results = cursor.fetchall()

    cursor.close()
    conn.close()
    
    return results

# Funzione per calcolare la media dei F1 score
def calculate_mean_f1(scores):
    llama_scores = []
    orca_scores = []
    falcon_scores = []

    # Dividi i risultati in base al modello
    for row in scores:
        f1_llama, f1_orca, f1_falcon = row
        llama_scores.append(f1_llama)
        orca_scores.append(f1_orca)
        falcon_scores.append(f1_falcon)
    
    # Calcola la media per ciascun modello
    mean_llama = sum(llama_scores) / len(llama_scores) if llama_scores else 0
    mean_orca = sum(orca_scores) / len(orca_scores) if orca_scores else 0
    mean_falcon = sum(falcon_scores) / len(falcon_scores) if falcon_scores else 0

    return mean_llama, mean_orca, mean_falcon

# Recupera i F1 score dal database
f1_scores = retrieve_f1_scores()

# Calcola la media dei F1 score per ciascun modello
mean_llama, mean_orca, mean_falcon = calculate_mean_f1(f1_scores)

# Stampa il risultato
print(f"Mean F1 Score for Llama: {mean_llama}")
print(f"Mean F1 Score for Orca: {mean_orca}")
print(f"Mean F1 Score for Falcon: {mean_falcon}")
