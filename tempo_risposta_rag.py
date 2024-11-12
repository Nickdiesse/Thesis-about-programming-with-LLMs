import psycopg2

# Connessione al database PostgreSQL
try:
    conn = psycopg2.connect(
        host="localhost",    
        database="benchmarkllm",  
        user="postgres",    
        password="nicola"  
    )

    cursor = conn.cursor()

    # Query per ottenere i tempi di risposta dei modelli
    query = """
    SELECT falcon_time, llama_time, orca_time 
    FROM model_rag_se
    """  

    cursor.execute(query)

    # Recupera i risultati dalla query
    risultati = cursor.fetchall()

    # Inizializza le variabili per calcolare la somma dei tempi e il numero di domande
    falcon_somma, llama3_somma, orca_somma = 0, 0, 0
    total_rows = len(risultati)

    # Somma i tempi di risposta per ogni modello
    for riga in risultati:
        falcon_somma += riga[0]  # Prima colonna per Falcon
        llama3_somma += riga[1]  # Seconda colonna per Llama3
        orca_somma += riga[2]    # Terza colonna per Orca

    # Calcola la media per ciascun modello
    falcon_media = falcon_somma / total_rows
    llama3_media = llama3_somma / total_rows
    orca_media = orca_somma / total_rows

    # Trova il modello con la media di tempo minore
    modello_più_rapido = min(
        ('Falcon', falcon_media),
        ('Llama3', llama3_media),
        ('Orca', orca_media),
        key=lambda x: x[1]
    )

    # Stampa i risultati
    print(f"Media tempi di risposta (secondi):")
    print(f"Falcon: {falcon_media:.2f}")
    print(f"Llama3: {llama3_media:.2f}")
    print(f"Orca: {orca_media:.2f}")
    print(f"Il modello più veloce è {modello_più_rapido[0]} con una media di {modello_più_rapido[1]:.2f} secondi.")

except Exception as e:
    print(f"Errore: {e}")

finally:
    if conn:
        cursor.close()
        conn.close()
