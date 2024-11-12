import json
import psycopg2

# Leggi il file JSON
with open(r'C:\Users\nicol\Desktop\progetto_tesi\benchmark\animal_questions.json') as file:
    data = json.load(file)

# Estrai le prime 30 domande e i loro id
domande = [(d['id'], d['text']) for d in data[:30]]

# Verifica il contenuto delle domande (opzionale)
print(domande)

# Connetti a PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="benchmarkllm",
    user="postgres",
    password="nicola"
)

# Crea un cursore
cur = conn.cursor()

# Scrivi la query di aggiornamento per ciascuna domanda
for domanda_id, domanda_text in domande:
    query = """
    UPDATE model_rag_responses
    SET question_text = %s
    WHERE question_id = %s;
    """
    cur.execute(query, (domanda_text.strip(), domanda_id))  # Usa l'id effettivo e rimuovi eventuali spazi iniziali/finali

# Conferma le modifiche
conn.commit()

# Chiudi la connessione
cur.close()
conn.close()
