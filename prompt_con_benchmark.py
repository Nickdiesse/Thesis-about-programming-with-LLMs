from gpt4all import GPT4All
import json
import statistics
import psycopg2


# Impostare la connessione al database
def connect_to_db():
    return psycopg2.connect(
        host="localhost",  # Inserisci nome host
        database="benchmarkllm",  # Inserisci il nome del database
        user="postgres",  # Inserisci il nome utente
        password="nicola"  # Inserisci la password
    )

# 1. Carica i modelli locali
model_orca = GPT4All(r"C:\Users\nicol\gpt4all\resources\orca-2-7b.Q4_0.gguf")
model_llama = GPT4All(r"C:\Users\nicol\gpt4all\resources\Meta-Llama-3-8B-Instruct.Q4_0.gguf")
model_falcon = GPT4All(r"C:\Users\nicol\gpt4all\resources\gpt4all-falcon-newbpe-q4_0.gguf")

# 2. Carica il benchmark SQUAD locale
with open(r"C:\Users\nicol\Desktop\progetto_tesi\benchmark\dev-v1.1.json") as f:
    squad_data = json.load(f)

# 3. Estrai le prime 100 domande
squad_subset = []
for entry in squad_data['data']:
    for paragraph in entry['paragraphs']:
        for qa in paragraph['qas']:
            squad_subset.append({
                'context': paragraph['context'],
                'question': qa['question'],
                'answer': qa['answers'][0]['text']
            })
            if len(squad_subset) >= 100:
                break
        if len(squad_subset) >= 100:
            break
    if len(squad_subset) >= 100:
        break

# 4. Funzione per fare domande ai modelli usando gpt4all
def ask_question_gpt4all(model, question, context):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = model.generate(prompt)
    return response

# 5. Metriche per il calcolo dell'Exact Match (EM) e F1
def exact_match(pred, truth):
    return int(pred.strip().lower() == truth.strip().lower())

def f1(pred, truth):
    pred_tokens = pred.split()
    truth_tokens = truth.split()
    common = set(pred_tokens) & set(truth_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

# 6. Funzione per salvare i risultati nel database
def save_to_db(cur, question, context, truth, answer_orca, answer_llama, answer_falcon, em_orca, f1_orca, em_llama, f1_llama, em_falcon, f1_falcon):
    cur.execute("""
        INSERT INTO benchmark_results (question, context, ground_truth, answer_orca, answer_llama, answer_falcon, em_orca, f1_orca, em_llama, f1_llama, em_falcon, f1_falcon)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """, (question, context, truth, answer_orca, answer_llama, answer_falcon, em_orca, f1_orca, em_llama, f1_llama, em_falcon, f1_falcon))

# 7. Connessione al database
conn = connect_to_db()
cur = conn.cursor()

try:
    # Itera sulle 100 domande del benchmark locale
    for idx, example in enumerate(squad_subset, 1):
        question = example['question']
        context = example['context']
        truth = example['answer']

        # Ottieni risposte dai modelli
        answer_orca = ask_question_gpt4all(model_orca, question, context)
        answer_llama = ask_question_gpt4all(model_llama, question, context)
        answer_falcon = ask_question_gpt4all(model_falcon, question, context)

        # Confronta le risposte con la verità fornita (truth)
        em_orca = exact_match(answer_orca, truth)
        f1_orca = f1(answer_orca, truth)

        em_llama = exact_match(answer_llama, truth)
        f1_llama = f1(answer_llama, truth)

        em_falcon = exact_match(answer_falcon, truth)
        f1_falcon = f1(answer_falcon, truth)

        # Salva i risultati nel database
        save_to_db(cur, question, context, truth, answer_orca, answer_llama, answer_falcon, em_orca, f1_orca, em_llama, f1_llama, em_falcon, f1_falcon)

        # Conferma l'inserimento dopo ogni iterazione
        conn.commit()

        # Stampa progressiva
        print(f"Salvati {idx} risultati su 100.")

    print("Dati salvati con successo nel database.")

except Exception as e:
    print(f"Errore durante il salvataggio dei dati: {e}")
    conn.rollback()

finally:
    # Chiudi la connessione
    cur.close()
    conn.close()
