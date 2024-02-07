import pandas as pd
from sentence_transformers import InputExample
import pandas as pd
from typing import List

def load_nli_samples(data_path: str, faq_pool: pd.DataFrame) -> List[InputExample]:
    """
    Load NLI samples from file
    """
    nli_samples = []
    
    data = pd.read_csv(data_path)

    for _, line in data.iterrows():
        line = line.strip()
        if line == "":
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        nli_samples.append(InputExample(texts=[parts[0], parts[1]], label=parts[2]))

    return nli_samples 