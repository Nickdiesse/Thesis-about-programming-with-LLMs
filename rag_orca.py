from gpt4all import GPT4All
from langchain_community.retrievers import WikipediaRetriever
from wikipediaapi import Wikipedia

# Caricamento del modello GPT4All
model_path = r'C:\Users\nicol\gpt4all\resources\orca-2-7b.Q4_0.gguf'
gpt4all_model = GPT4All(model_path)

class LocalLLM:
    def __init__(self, model):
        self.model = model

    def __call__(self, prompt):
        response = self.model.generate(prompt)
        return response

llm = LocalLLM(gpt4all_model)

# Configurazione del retriever di Wikipedia con un user agent appropriato
class WikipediaRetriever:
    def __init__(self):
        self.wiki = Wikipedia(language='en', user_agent= "RAG_project/1.0 (nicoladesiena@outlook.it)")

    def retrieve(self, query):
        page = self.wiki.page(query)
        if page.exists():
            return page.summary
        else:
            return "No relevant information found."

retriever = WikipediaRetriever()

# Configurazione della catena di RAG
class RetrievalQA:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def __call__(self, query):
        context = self.retriever.retrieve(query)
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        return self.llm(prompt)

qa_chain = RetrievalQA(llm, retriever)

# Esecuzione del sistema di QA
def main():
    while True:
        query = input("Ask a question about actors: ")
        if query.lower() in ['exit', 'quit']:
            break
        answer = qa_chain(query)
        print(f"Risposta: {answer}")

if __name__ == "__main__":
    main()
