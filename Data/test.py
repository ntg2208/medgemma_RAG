# %%
import logging

from preprocessing import DocumentPreprocessor
# %%
logging.basicConfig(level=logging.INFO)

preprocessor = DocumentPreprocessor()
documents = preprocessor.process_directory()

if documents:
    stats = preprocessor.get_document_stats(documents)
    print("\nDocument Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
# %%
print(len(documents))
#%%
print(documents[0]) 
