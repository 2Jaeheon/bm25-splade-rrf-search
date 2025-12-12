import ir_datasets
from collections import Counter
import re

def analyze_top_words(dataset_id="wikir/en1k/training", top_k=50):
    dataset = ir_datasets.load(dataset_id)
    
    counter = Counter()
    doc_count = 0
    
    for doc in dataset.docs_iter():
        text = doc.text.lower()
        tokens = re.findall(r'[a-z]+', text)
        counter.update(tokens)
        
        doc_count += 1
        if doc_count % 1000 == 0:
            print(f"Processed {doc_count} docs...")

    print(f"\nCompleted! Processed {doc_count} documents.")
    print(f"\nTop {top_k} most frequent words:")
    print("-" * 40)
    print(f"{'Rank':<5} | {'Word':<15} | {'Frequency':<10}")
    print("-" * 40)
    
    for rank, (word, freq) in enumerate(counter.most_common(top_k), 1):
        print(f"{rank:<5} | {word:<15} | {freq:<10}")
        
    return [word for word, freq in counter.most_common(top_k)]

if __name__ == "__main__":
    analyze_top_words()
