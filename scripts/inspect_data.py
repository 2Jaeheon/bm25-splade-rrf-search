import ir_datasets
from itertools import islice

# 데이터 셋 가져오기
dataset = ir_datasets.load("wikir/en1k/training")

print("문서 확인")
# 5개 문서 확인
for doc in islice(dataset.docs_iter(), 5):
    print(f"Doc ID: {doc.doc_id}")
    print(f"Text (truncated): {doc.text[:100]}...")
    print("-" * 30)

print("\n쿼리 확인")
# 5개 쿼리 확인
for query in islice(dataset.queries_iter(), 5):
    print(f"Query ID: {query.query_id}")
    print(f"Text: {query.text}")
    print("-" * 30)

print("\nRelevance 확인")
# 5개 qrels(query-document의 relevance) 확인
for qrel in islice(dataset.qrels_iter(), 5):
    print(f"Query ID: {qrel.query_id}, Doc ID: {qrel.doc_id}, Relevance: {qrel.relevance}")
    print("-" * 30)
