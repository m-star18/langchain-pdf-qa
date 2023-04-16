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
# Example
python src/make_emb.py --pdf_url "https://arxiv.org/pdf/2005.14165.pdf" --OPEN_API_KEY ""
```

3. Enter a question and generate an answer from the extracted text
```
# Example
python src/pdf_qa.py --query "On which datasets does GPT-3 struggle?"
```
