import os
import PyPDF2
from transformers import T5ForConditionalGeneration, T5Tokenizer

def summarize_pdfs(pdf_directory):
    files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    summaries = []

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    for fileX in files:
        with open(fileX, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''.join([page.extract_text() for page in reader.pages])

        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1000, truncation=True)
        outputs = model.generate(inputs, max_length=1000, min_length=100, length_penalty=3.0, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(outputs[0])
        summaries.append(summary)

    return summaries
