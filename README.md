# langchain-pdf-qa

Use langchain to create a model that returns answers based on online PDFs that have been read.

## How to use

1. Clone the repository and install dependencies
```
git clone git@github.com:m-star18/langchain-pdf-qa.git
pip install -r requirements.txt
```

2. Specify the PDF link and OPEN_API_KEY to create the embedding model
```
python src/make_emb.py
```

3. Enter a question and generate an answer from the extracted text
```
python src/pdf_qa.py
```
