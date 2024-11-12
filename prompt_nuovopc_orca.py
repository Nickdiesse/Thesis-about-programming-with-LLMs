# Importazione delle librerie necessarie
from gpt4all import GPT4All


# Percorso al file del modello scaricato
model_path = r'C:\Users\nicol\gpt4all\resources\orca-2-7b.Q4_0.gguf'

# Inizializzazione del modello
gpt4all_model = GPT4All(model_path)


# Funzione per creare il prompt
def create_prompt(question):
    return f"Answer this question about Formula 1: {question}"

# Funzione per ottenere la risposta
def answer_question(question):
    prompt = create_prompt(question)
    response = gpt4all_model.generate(prompt)
    return response

# Esempi di domande
questions = [
    "who won the formula 1 world championship in 2023?",
    "what is the longest formula 1 track currently ?",
    "How many championships has Lewis Hamilton won?"
]

# Esecuzione dei test
for question in questions:
    answer = answer_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
