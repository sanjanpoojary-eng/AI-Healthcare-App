import io
from PyPDF2 import PdfReader

def extract_text(file_bytes):
    text = ""
    with io.BytesIO(file_bytes) as pdf_file:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text