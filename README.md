### Overview

This project demonstrates how to build a chatbot using a Retrieval-Augmented Generation (RAG) model with Azure OpenAI for language processing and a local document vector database for document retrieval. The chatbot can handle various document types (PDF, PPTX, DOCX, TXT) and supports interactive user sessions with a chat history.

### Requirements

- Python 3.8 or higher
- Streamlit
- OpenAI
- PyMuPDF (fitz)
- pdf2image
- Pillow (PIL)
- python-pptx
- docx2txt
- sentence-transformers
- faiss
- pickle
- numpy
- logging

### Installation

Install the required libraries using pip:

```bash
pip install streamlit openai pymupdf pdf2image pillow python-pptx docx2txt sentence-transformers faiss-cpu

```

### Directory Structure

The following directories are required:

- `documents/` for storing uploaded documents.
- `vector_store/` for storing the FAISS index and document metadata.

Create these directories if they do not exist:

```bash
mkdir -p documents vector_store

```

### Code Explanation

### 1. **Configuration**

The script starts by importing the necessary libraries and setting up the logging configuration.

### 2. **Secrets Manager**

A simple secrets manager retrieves API keys and endpoint information:

```python
class SecretsManager:
    @staticmethod
    def get_secret(secret_name):
        secrets = {
            "AZURE_API_KEY": "your-azure-api-key",
            "AZURE_API_DEPLOYMENT": "your-deployment-id",
            "AZURE_API_ENDPOINT": "your-endpoint"
        }
        return secrets.get(secret_name)

```

Replace `"your-azure-api-key"`, `"your-deployment-id"`, and `"your-endpoint"` with your actual Azure OpenAI credentials.

### 3. **Azure OpenAI Configuration**

Configure Azure OpenAI with the retrieved secrets:

```python
key = SecretsManager.get_secret('AZURE_API_KEY')
deployment = SecretsManager.get_secret("AZURE_API_DEPLOYMENT")
endpoint = SecretsManager.get_secret("AZURE_API_ENDPOINT")

openai.api_type = "azure"
openai.api_key = key
openai.api_base = endpoint
openai.api_version = "2024-02-01"

```

### 4. **Sentence Transformer Model**

Initialize the Sentence Transformer model for embedding documents:

```python
model = SentenceTransformer('all-MiniLM-L6-v2')

```

### 5. **FAISS Index and Metadata**

Load or create a FAISS index and metadata for document embeddings:

```python
if os.path.exists('vector_store/faiss_index'):
    with open('vector_store/faiss_index', 'rb') as f:
        index = pickle.load(f)
else:
    index = faiss.IndexIDMap(faiss.IndexFlatL2(384))

metadata_path = 'vector_store/doc_metadata.pkl'
if os.path.exists(metadata_path):
    with open(metadata_path, 'rb') as f:
        doc_metadata = pickle.load(f)
else:
    doc_metadata = {}

```

### 6. **Text Extraction**

Extract text from various document types:

```python
def extract_text_from_doc(file_path):
    if file_path.endswith('.pdf'):
        text = ""
        pdf = fitz.open(file_path)
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            text += page.get_text()
        return text
    elif file_path.endswith('.pptx'):
        prs = Presentation(file_path)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text
        return text
    elif file_path.endswith('.docx'):
        return docx2txt.process(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            return file.read()
    else:
        return ""

```

### 7. **Embedding Documents**

Embed documents and update the FAISS index and metadata:

```python
def embed_documents(file_paths):
    def split_text(text, max_length=512):
        paragraphs = text.split('\\n\\n')
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) < max_length:
                current_chunk += para + "\\n\\n"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = para + "\\n\\n"
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    for file_path in file_paths:
        text = extract_text_from_doc(file_path)
        if text:
            chunks = split_text(text)
            embeddings = model.encode(chunks)
            ids = np.arange(index.ntotal, index.ntotal + len(embeddings))
            index.add_with_ids(np.array(embeddings, dtype=np.float32), ids)
            for i, chunk in enumerate(chunks):
                doc_metadata[int(ids[i])] = {
                    'text': chunk,
                    'file_path': file_path
                }
            with open('vector_store/faiss_index', 'wb') as f:
                pickle.dump(index, f)
            with open(metadata_path, 'wb') as f:
                pickle.dump(doc_metadata, f)

```

### 8. **Handle Document Uploads**

Handle the document uploads and trigger embedding:

```python
def handle_upload(uploaded_files):
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join('documents', uploaded_file.name)
        with open(file_path, 'wb') as file:
            file.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    embed_documents(file_paths)
    return file_paths

```

### 9. **AzureChatOpenAI Class**

Define a class for interacting with Azure OpenAI:

```python
class AzureChatOpenAI:
    def __init__(self, model_name, temperature, openai_api_key, azure_deployment, azure_endpoint, openai_api_version):
        self.model_name = model_name
        self.temperature = temperature
        self.openai_api_key = openai_api_key
        self.azure_deployment = azure_deployment
        self.azure_endpoint = azure_endpoint
        self.openai_api_version = openai_api_version

    def generate_response(self, prompt):
        response = openai.ChatCompletion.create(
            deployment_id=self.azure_deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()

```

### 10. **Chatbot Response Generation**

Generate responses using Azure OpenAI and document embeddings:

```python
def get_chatbot_response(user_input):
    user_embedding = model.encode([user_input])
    D, I = index.search(np.array(user_embedding, dtype=np.float32), 5)
    relevant_docs = []
    for idx in I[0]:
        if idx != -1:
            doc_data = doc_metadata.get(int(idx), None)
            if doc_data:
                relevant_docs.append(doc_data['text'])

    max_tokens = 1000
    prompt_docs = []
    current_tokens = 0
    for doc in relevant_docs:
        tokens = len(doc.split())
        if current_tokens + tokens > max_tokens:
            break
        prompt_docs.append(doc)
        current_tokens += tokens

    chat_history = get_previous_chat_history()
    history_text = "\\n".join([f"User: {entry['question']}\\nAssistant: {entry['answer']}" for entry in chat_history])
    prompt = f"{history_text}\\nUser asked: {user_input}\\nRelevant documents: {' '.join(prompt_docs)}\\nAnswer:"
    response = chat_model.generate_response(prompt)
    save_chat_history(user_input, response)
    return response

```

### 11. **Terminate Program**

Terminate the program and close the browser:

```python
def terminate_program():
    st.write("Shutting down...")
    os.kill(os.getpid(), signal.SIGTERM)

```

### 12. **Streamlit UI**

Create the Streamlit user interface:

```python
def streamlit_ui():
    st.title("RAG Model with Azure OpenAI and Local Documents")

    st.write("Upload documents for the RAG model.")
    uploaded_files = st.file_uploader("Choose files", type=['pdf', 'pptx', 'docx', 'txt'], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner('Uploading and processing...'):
            file_paths = handle_upload(uploaded_files)
            if file_paths:
                for file_path in file_paths:
                    st.success(f"Uploaded {os.path.basename(file_path)}")
            else:
                st.error("Failed to upload files. Check logs for details.")

    st.write("Ask a question to the chatbot:")
    user_input = st.text_input("Your question:")
    if st.button("Ask"):
        if user_input:
            with st.spinner('Processing...'):
                response = get_chatbot_response(user_input)
                if response:
                    st.write("Chatbot:", response)
                else:
                    st.error("Failed to get a response. Check logs for details.")

    st.write("Chat History:")
    chat_history = get_previous_chat_history()
    for entry in chat_history:
        st.write(f"User: {entry['question']}")
        st.write(f"Assistant: {entry['answer']}")

    if st.button("Stop Chat"):
        terminate_program()

if __name__ == "__main__":
    streamlit_ui()

```

### Running the Chatbot

Run the Streamlit application with the following command:

```bash
streamlit run chatbot_app.py

```

This will open a web interface where you can upload documents, ask questions, and interact with the chatbot. Pressing the "Stop Chat" button will terminate the program and close the browser.

### Summary

This project provides an end-to-end solution for building a RAG model chatbot using Azure OpenAI and local document embeddings. It includes document handling, embedding, vector database management, and a Streamlit-based user interface. The chatbot retains session history for context-aware responses and allows for a clean termination of the program through the UI.
