import sys
import os
import pytrec_eval
import ir_datasets
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.search_engine import SearchEngine

def main():
    # 엔진 및 데이터셋 로드
    engine = SearchEngine(index_path="data/index.pkl")
    if not engine.load():
        return
    dataset_id = "wikir/en1k/training"
    dataset = ir_datasets.load(dataset_id)
    
    qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.query_id not in qrels:
            qrels[qrel.query_id] = {}
        
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    queries = {}
    for query in dataset.queries_iter():
        queries[query.query_id] = query.text
    
    # 실제 평가 실행
    run = {}
    target_query_ids = set(qrels.keys())
    for q_id, q_text in tqdm(queries.items(), desc="검색 중"):
        if q_id not in target_query_ids:
            continue

        results = engine.search_bm25(q_text, top_k=5000)
        
        run[q_id] = {}
        for doc_id, score in results:
            run[q_id][doc_id] = score

    # 평가 지표
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, 
        {'map', 'ndcg', 'P_10', 'recall_100', 'recall_1000', 'recall_2000', 'recall_5000'}
    )
    metrics = evaluator.evaluate(run)
    
    aggregated = {
        'map': 0.0, 'ndcg': 0.0, 'P_10': 0.0, 
        'recall_100': 0.0, 'recall_1000': 0.0, 
        'recall_2000': 0.0, 'recall_5000': 0.0
    }
    
    # 점수 집계
    count = len(metrics)
    for q_id, scores in metrics.items():
        for measure in aggregated.keys():
            aggregated[measure] += scores.get(measure, 0.0)
            
    if count > 0:
        for measure in aggregated:
            aggregated[measure] /= count

    print("\n" + "="*30)
    print("           평가 결과")
    print("="*30)
    print(f"MAP:            {aggregated['map']:.4f}")
    print(f"nDCG:           {aggregated['ndcg']:.4f}")
    print(f"P@10:           {aggregated['P_10']:.4f}")
    print(f"Recall@100:     {aggregated['recall_100']:.4f}")
    print(f"Recall@1000:    {aggregated['recall_1000']:.4f}")
    print(f"Recall@2000:    {aggregated['recall_2000']:.4f}")
    print(f"Recall@5000:    {aggregated['recall_5000']:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
