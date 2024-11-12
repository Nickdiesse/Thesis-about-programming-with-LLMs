from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
import numpy as np

# Carica il modello di embedding
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Funzione per connettersi al database e recuperare le risposte
def retrieve_responses():
    conn = psycopg2.connect(
        host="localhost",
        database="benchmarkllm",
        user="postgres",
        password="nicola"
    )
    cursor = conn.cursor()

    # Recupera tutte le risposte dal database, inclusa la ground truth
    cursor.execute("SELECT id, question, ground_truth, answer_llama, answer_orca, answer_falcon FROM benchmark_results")
    results = cursor.fetchall()

    cursor.close()
    conn.close()
    
    return results

# Funzione per calcolare embedding e confrontare risposte
def compare_responses_with_embeddings(results):
    # Contatori per tracciare quante volte ciascun modello è il migliore
    best_model_correctness_count = {
        "Llama": 0,
        "Orca": 0,
        "Falcon": 0
    }
    
    best_model_semantic_count = {
        "Llama": 0,
        "Orca": 0,
        "Falcon": 0
    }

    comparison = []
    
    for row in results:
        q_id, q_text, ground_truth, answer_llama, answer_orca, answer_falcon = row
        
        # Calcola embedding per ogni risposta e per la ground truth
        embedding_llama = embedding_model.encode(answer_llama)
        embedding_orca = embedding_model.encode(answer_orca)
        embedding_falcon = embedding_model.encode(answer_falcon)
        embedding_ground_truth = embedding_model.encode(ground_truth)
        
        # Misura la lunghezza dei vettori (ricchezza semantica)
        length_llama = np.linalg.norm(embedding_llama)
        length_orca = np.linalg.norm(embedding_orca)
        length_falcon = np.linalg.norm(embedding_falcon)
        
        # Verifica quanto le risposte siano simili alla ground truth usando cosine similarity
        similarity_llama = cosine_similarity([embedding_llama], [embedding_ground_truth])[0][0]
        similarity_orca = cosine_similarity([embedding_orca], [embedding_ground_truth])[0][0]
        similarity_falcon = cosine_similarity([embedding_falcon], [embedding_ground_truth])[0][0]
        
        # Imposta una soglia per considerare una risposta "corretta"
        similarity_threshold = 0.8  # Valore arbitrario, si può sperimentare con diversi valori
        
        # Step 1: Controlla se le risposte sono corrette rispetto alla soglia di similarità
        correct_models = []
        if similarity_llama >= similarity_threshold:
            correct_models.append("Llama")
            best_model_correctness_count["Llama"] += 1
        
        if similarity_orca >= similarity_threshold:
            correct_models.append("Orca")
            best_model_correctness_count["Orca"] += 1
            
        if similarity_falcon >= similarity_threshold:
            correct_models.append("Falcon")
            best_model_correctness_count["Falcon"] += 1
        
        # Step 2: Tra i modelli corretti, determina quale ha la migliore ricchezza semantica
        if correct_models:
            best_model = None
            max_length = 0
            
            if "Llama" in correct_models and length_llama > max_length:
                best_model = "Llama"
                max_length = length_llama
            
            if "Orca" in correct_models and length_orca > max_length:
                best_model = "Orca"
                max_length = length_orca
            
            if "Falcon" in correct_models and length_falcon > max_length:
                best_model = "Falcon"
                max_length = length_falcon
            
            if best_model:
                best_model_semantic_count[best_model] += 1
                print(f"Question ID: {q_id} - Best Model by Semantic Richness (among correct ones): {best_model}")
                print(f"Llama Length: {length_llama}, Orca Length: {length_orca}, Falcon Length: {length_falcon}")
                print(f"Llama Similarity: {similarity_llama:.2f}, Orca Similarity: {similarity_orca:.2f}, Falcon Similarity: {similarity_falcon:.2f}")
        else:
            print(f"No correct model for Question ID {q_id}")
        
        print()
    
    # Stampa i risultati finali: quale modello ha risposto correttamente più volte
    print("\nCorrectness Results:")
    for model, count in best_model_correctness_count.items():
        print(f"{model} was correct {count} times.")
    
    # Stampa i risultati finali: quale modello ha la migliore semantica tra quelli corretti
    print("\nSemantic Richness Results (among correct models):")
    for model, count in best_model_semantic_count.items():
        print(f"{model} was the best model by semantic richness {count} times.")
    
    return comparison

# Recupera le risposte dal database
results = retrieve_responses()

# Confronta le risposte usando embedding e similarità
comparison = compare_responses_with_embeddings(results)
