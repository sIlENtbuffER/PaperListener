python -m venv PaperListener
source PaperListener/bin/activate
pip install pdfminer.six regex requests pymupdf fitz
python CleanPaper.py "Your url || Absolute Path || File Name in Data"