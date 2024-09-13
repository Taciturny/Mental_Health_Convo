import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from src.core.config import settings
from src.core.search_engine import SearchEngine

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QdrantRetrievalEvaluator:
    def __init__(self, search_engine: SearchEngine, ground_truth_file: str):
        self.search_engine = search_engine
        self.ground_truth = self.load_ground_truth(ground_truth_file)

    @staticmethod
    def load_ground_truth(file_path: str) -> Dict[str, str]:
        """Load ground truth data from a parquet file"""
        df = pd.read_parquet(file_path)
        return dict(zip(df["id"], df["question"]))

    def evaluate(self, k: int = 5):
        search_types = ["dense", "late", "hybrid"]
        results = defaultdict(lambda: defaultdict(list))

        for search_type in search_types:
            mlflow.set_experiment(
                f"Qdrant Retrieval Evaluation - {search_type.capitalize()}"
            )

            with mlflow.start_run():
                mlflow.log_param("k", k)
                mlflow.log_param("search_type", search_type)

                for query_id, query_text in self.ground_truth.items():
                    if search_type == "dense":
                        search_results = self.search_engine.search_dense(query_text)
                    elif search_type == "late":
                        search_results = self.search_engine.search_late(query_text)
                    else:  # hybrid
                        search_results = self.search_engine.search_hybrid(query_text)

                    relevant_docs = set([query_id])

                    retrieved_docs = [point.id for point in search_results.points[:k]]
                    retrieved_scores = [
                        point.score for point in search_results.points[:k]
                    ]

                    metrics = self.calculate_metrics(
                        retrieved_docs, retrieved_scores, relevant_docs, k
                    )
                    for metric, value in metrics.items():
                        results[search_type][metric].append(value)

                mean_metrics = {
                    metric: np.mean(values)
                    for metric, values in results[search_type].items()
                }
                for metric, value in mean_metrics.items():
                    mlflow.log_metric(f"mean_{metric}", value)

        self.save_results_to_csv(results, k)
        return results

    def calculate_metrics(
        self,
        retrieved_docs: List[str],
        retrieved_scores: List[float],
        relevant_docs: set,
        k: int,
    ) -> Dict[str, float]:
        precision = self.precision_at_k(retrieved_docs, relevant_docs, k)
        recall = self.recall_at_k(retrieved_docs, relevant_docs)
        mrr = self.mrr(retrieved_docs, relevant_docs)
        ndcg = self.ndcg(retrieved_docs, relevant_docs, retrieved_scores, k)
        ap = self.average_precision(retrieved_docs, relevant_docs)
        f1 = self.f1_score(precision, recall)
        r_precision = self.r_precision(retrieved_docs, relevant_docs)

        return {
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
            "ndcg": ndcg,
            "map": ap,
            "f1": f1,
            "r_precision": r_precision,
        }

    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: set, k: int) -> float:
        return len(set(retrieved_docs[:k]) & relevant_docs) / k

    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: set) -> float:
        return len(set(retrieved_docs) & relevant_docs) / len(relevant_docs)

    @staticmethod
    def mrr(retrieved_docs: List[str], relevant_docs: set) -> float:
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1 / (i + 1)
        return 0

    @staticmethod
    def ndcg(
        retrieved_docs: List[str],
        relevant_docs: set,
        scores: List[float],
        k: int,
    ) -> float:
        relevance = [1 if doc in relevant_docs else 0 for doc in retrieved_docs[:k]]
        return ndcg_score([relevance], [scores], k=k)

    @staticmethod
    def average_precision(retrieved_docs: List[str], relevant_docs: set) -> float:
        score = 0.0
        num_hits = 0.0
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score / len(relevant_docs) if relevant_docs else 0.0

    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        return (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

    @staticmethod
    def r_precision(retrieved_docs: List[str], relevant_docs: set) -> float:
        r = len(relevant_docs)
        return len(set(retrieved_docs[:r]) & relevant_docs) / r if r > 0 else 0

    def save_results_to_csv(self, results: Dict[str, Dict[str, List[float]]], k: int):
        filename = f"retrieval_evaluation_results_k{k}.csv"
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            headers = ["Search Type"] + list(next(iter(results.values())).keys())
            writer.writerow(headers)

            for search_type, metrics in results.items():
                row = [search_type] + [
                    f"{np.mean(values):.4f}" for values in metrics.values()
                ]
                writer.writerow(row)

        print(f"Results saved to {filename}")


def main():
    # Initialize your SearchEngine
    search_engine = SearchEngine(settings.COLLECTION_NAME_LOCAL)

    # Path to your ground truth parquet file
    ground_truth_file = settings.GROUND_TRUTH_DATA

    # Initialize the evaluator
    evaluator = QdrantRetrievalEvaluator(search_engine, ground_truth_file)

    # Run the evaluation
    k = 5
    results = evaluator.evaluate(k=k)

    # Print the results
    print(f"Evaluation Results (k={k}):")
    for search_type, metrics in results.items():
        print(f"\n{search_type.capitalize()} Search:")
        for metric, values in metrics.items():
            print(f"  {metric.upper()}: {np.mean(values):.4f}")


if __name__ == "__main__":
    main()
