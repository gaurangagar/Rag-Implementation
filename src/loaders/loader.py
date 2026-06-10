from langchain_community.document_loaders import CSVLoader, DirectoryLoader, JSONLoader, PyPDFLoader, TextLoader
from typing import List, Any
from pathlib import Path

def load_documents(file_path:str)->List[Any]:
    """
    Load all supported files into langchain document structures.
    Supported: CSV, Directory, JSON, PDF, TEXT
    
    Args:
        file_path(str): The path to the file or directory.
        
    Returns:
        List[Any]: A list of loaded documents.
    """
    path=Path(file_path).resolve()
    print(path)
    documents=[]

    ##pdf files
    pdf_files=list(path.glob('**/*.pdf'))
    for file in pdf_files:
        try:
            loader=PyPDFLoader(str(file))
            loaded=loader.load()
            documents.extend(loaded)
        except Exception as e:
            print(f"Error loading PDF file {file}: {e}")

    ##text files
    text_files=list(path.glob('**/*.txt'))
    for file in text_files:
        try:
            loader=TextLoader(str(file))
            loaded=loader.load()
            documents.extend(loaded)
        except Exception as e:
            print(f"Error loading text file {file}: {e}")

    ##csv files
    csv_files=list(path.glob('**/*.csv'))
    for file in csv_files:
        try:
            loader=CSVLoader(str(file))
            loaded=loader.load()
            documents.extend(loaded)
        except Exception as e:
            print(f"Error loading CSV file {file}: {e}")

    ##json files
    json_files=list(path.glob('**/*.json'))
    for file in json_files:
        try:
            loader=JSONLoader(str(file))
            loaded=loader.load()
            documents.extend(loaded)
        except Exception as e:
            print(f"Error loading JSON file {file}: {e}")
    
    return documents

if __name__=='__main__':
    docs=load_documents('data')
    print(len(docs))
    print(docs)