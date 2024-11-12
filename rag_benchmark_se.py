from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer
import os
import time
import json
from gpt4all import GPT4All
import psycopg2



def create_table():
    conn = psycopg2.connect(
        host="localhost",
        database="benchmarkllm",
        user="postgres",
        password="nicola"
    )
    cursor = conn.cursor()

 # Creazione tabella (se non esiste)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_rag_responses_se (
    question_id INT,
    question_text TEXT,
    answer_llama TEXT,
    answer_orca TEXT,
    answer_falcon TEXT,
    llama_time FLOAT,
    orca_time FLOAT,
    falcon_time FLOAT
    );
    """)
    conn.commit()
    cursor.close()
    conn.close()

# Funzione per salvare gradualmente le risposte nel database
def save_to_db(result):
    conn = psycopg2.connect(
        host="localhost",
        database="benchmarkllm",
        user="postgres",
        password="nicola"
    )
    cursor = conn.cursor()

    # Inserimento dati nel database
    cursor.execute("""
        INSERT INTO model_rag_responses_se (question_id, question_text, answer_llama, answer_orca, answer_falcon, llama_time, orca_time, falcon_time)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """, (
        result["question_id"],
        result["question_text"],
        result["answer_llama"],
        result["answer_orca"],
        result["answer_falcon"],
        result["llama_time"],
        result["orca_time"],
        result["falcon_time"]
    ))

    # Esegui il commit e chiudi la connessione
    conn.commit()
    cursor.close()
    conn.close()






# Caricamento dei PDF e creazione di un indice di vettori
def create_pdf_index(pdf_folder):
    loaders = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loaders.append(PyPDFLoader(os.path.join(pdf_folder, file)))
    
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = FAISS.from_documents(documents, embeddings)
    return index

pdf_index = create_pdf_index(r"C:\Users\nicol\Desktop\progetto_tesi\benchmark\pdf_software_engineering")

# Caricamento delle domande dal file JSON
with open(r"C:\Users\nicol\Desktop\progetto_tesi\benchmark\software_engineering.json") as f:
    questions = json.load(f)

# Carica i modelli .gguf
models = {
    "Llama": GPT4All(r"C:\Users\nicol\gpt4all\resources\Meta-Llama-3-8B-Instruct.Q4_0.gguf"),
    "Orca": GPT4All(r"C:\Users\nicol\gpt4all\resources\orca-2-7b.Q4_0.gguf"),
    "Falcon": GPT4All(r"C:\Users\nicol\gpt4all\resources\gpt4all-falcon-newbpe-q4_0.gguf")
}



# Carica il modello e il tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")



def ask_question_to_models(models, question, pdf_index):
    
    context = pdf_index.similarity_search(question, k=3)
    context_text = " ".join([doc.page_content for doc in context])

    # Tokenizza il contesto per verificare la lunghezza
    context_tokens = tokenizer(context_text)["input_ids"]
    
    
    
    # Limita il numero di token totali (per esempio, 2048 - lunghezza della domanda)
    max_context_tokens = 1800  # Lascia un margine per la domanda
    if len(context_tokens) > max_context_tokens:
        # Ritaglia il contesto per non eccedere i 1800 token
        context_text = tokenizer.decode(context_tokens[:max_context_tokens])
    
    # Fai domanda a ogni modello
    responses = {}
    response_times = {}  # Dizionario per salvare i tempi di risposta
    for model_name, model in models.items():
        prompt = f"Context: {context_text}\nQuestion: {question}\nAnswer:"
        
        # Misurazione del tempo di inizio
        start_time = time.time()
        
        # Utilizza il metodo generate per ottenere la risposta
        response = model.generate(prompt)
        
        # Misurazione del tempo di fine
        end_time = time.time()
        
        # Calcola il tempo di risposta
        response_time = end_time - start_time
        response_times[model_name] = response_time
        
        # Verifica se la risposta è una stringa o un dizionario
        if isinstance(response, str):
            responses[model_name] = response
        elif isinstance(response, dict) and "choices" in response:
            responses[model_name] = response["choices"][0]["text"]
        else:
            responses[model_name] = "Error: Unexpected response format"
        
        # Stampa il progresso della risposta e il tempo impiegato
        print(f"Model: {model_name} - Question: {question[:30]}... - Response: {responses[model_name][:50]}... - Time: {response_time:.2f} seconds")
    
    return responses, response_times  # Ritorna anche i tempi di risposta




create_table()  # Creazione della tabella nel database

for question in questions:
    q_id = question["id"]
    q_text = question["text"]
    print(f"Processing Question ID: {q_id} - {q_text[:50]}...")  # Messaggio di inizio per ogni domanda
    
    # Ottieni le risposte dai modelli e i tempi di risposta
    responses, response_times = ask_question_to_models(models, q_text, pdf_index)
    
    # Prepara il dizionario per salvare i risultati di ogni modello
    result = {
        "question_id": q_id,
        "question_text": q_text,
        "answer_llama": responses.get("Llama", ""),
        "answer_orca": responses.get("Orca", ""),
        "answer_falcon": responses.get("Falcon", ""),
        "llama_time": response_times.get("Llama", 0),
        "orca_time": response_times.get("Orca", 0),
        "falcon_time": response_times.get("Falcon", 0)
    }
    
    # Salva la risposta e i tempi nel database per la domanda corrente
    save_to_db(result)
    print(f"Saved response and times for question ID {q_id} to the database.")  