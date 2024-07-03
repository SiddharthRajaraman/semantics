from PyPDF2 import PdfReader

def text_extraction_pypdf(pdf_path):
    reader = PdfReader(pdf_path)

    # print(len(reader.pages))

    text = ""
    for i in range(len(reader.pages)):
        text += reader.pages[i].extract_text()


    # text = reader.pages[0].extract_text()

    # print(text)
    return(text)



# text_extraction_pypdf()