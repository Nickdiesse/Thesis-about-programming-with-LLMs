import json
import psycopg2


with open(r"C:\Users\nicol\Desktop\progetto_tesi\Rag_con_db\champions_league.json") as file:
    data = json.load(file)



# Connessione a PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="benchmarkllm",
    user="postgres",
    password="nicola"
)
cursor = conn.cursor()

# Aggiungi una nuova colonna ground_truth alla tabella
try:
    cursor.execute("ALTER TABLE qa_results ADD COLUMN ground_truth TEXT;")
    conn.commit()
except psycopg2.errors.DuplicateColumn:
    conn.rollback()  # Ignora l'errore se la colonna esiste già

# Aggiorna la tabella con i valori 'answer' del file JSON
for item in data:
    answer_value = item["answer"]
    
    # Converti l'array in stringa per gestire risposte multiple
    if isinstance(answer_value, list):
        answer_value = "; ".join(answer_value)
    
    # Esegui l'aggiornamento nella tabella
    cursor.execute(
        "UPDATE qa_results SET ground_truth = %s WHERE id = %s;",
        (answer_value, item["id"])
    )

# Conferma le modifiche e chiudi la connessione
conn.commit()
cursor.close()
conn.close()
