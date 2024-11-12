import psycopg2
import re

def calculate_f1_and_exact(predicted, ground_truth):
    # Tokenizza il testo dividendo su spazi e caratteri speciali
    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    pred_tokens = tokenize(predicted)
    gt_tokens = tokenize(ground_truth)

    # Calcola l'F1-score
    common_tokens = set(pred_tokens) & set(gt_tokens)
    num_common = len(common_tokens)

    if num_common == 0:
        f1 = 0.0
    else:
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)

    # Calcola l'Exact Match
    exact_match = int(predicted.strip().lower() == ground_truth.strip().lower())

    return f1, exact_match

# Connessione al database PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="benchmarkllm",
    user="postgres",
    password="nicola"
)
cursor = conn.cursor()

# Crea una nuova tabella per salvare i risultati
cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_evaluation (
        answer_id INT PRIMARY KEY,
        f1_llama DOUBLE PRECISION,
        f1_orca DOUBLE PRECISION,
        f1_falcon DOUBLE PRECISION,
        em_llama INT,
        em_orca INT,
        em_falcon INT
    );
""")
conn.commit()

# Estrai i dati dalla tabella esistente
cursor.execute("SELECT id, ground_truth, answer_llama, answer_orca, answer_falcon FROM qa_results;")
rows = cursor.fetchall()

# Calcola F1-score ed Exact Match per ogni modello e salva nella nuova tabella
for row in rows:
    answer_id = row[0]
    ground_truth = row[1]
    answers = {
        "llama": row[2],
        "orca": row[3],
        "falcon": row[4]
    }

    # Calcola F1-score ed Exact Match per ciascun modello
    f1_llama, em_llama = calculate_f1_and_exact(answers["llama"], ground_truth)
    f1_orca, em_orca = calculate_f1_and_exact(answers["orca"], ground_truth)
    f1_falcon, em_falcon = calculate_f1_and_exact(answers["falcon"], ground_truth)

    # Inserisci i risultati nella tabella model_evaluation
    cursor.execute("""
        INSERT INTO model_evaluation (answer_id, f1_llama, f1_orca, f1_falcon, em_llama, em_orca, em_falcon)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (answer_id) DO NOTHING;
    """, (answer_id, f1_llama, f1_orca, f1_falcon, em_llama, em_orca, em_falcon))

# Conferma le modifiche e chiudi la connessione
conn.commit()
cursor.close()
conn.close()
