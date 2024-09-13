import sys
from pathlib import Path
from typing import Dict, List

import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.config import settings
from src.core.embeddings_model import EmbeddingsModel
from src.core.llm_model import EnsembleModel
from src.core.search_engine import SearchEngine

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def load_datasets():
    original_data = (
        pd.read_parquet(settings.DATA_FILE_PATH)
        .sample(n=4000, random_state=42)
        .to_dict(orient="records")
    )  # use a sample size of 4000 or less to save time
    ground_truth = pd.read_parquet(settings.GROUND_TRUTH_DATA).to_dict(orient="records")
    return original_data, ground_truth


def perform_dense_search(
    search_engine: SearchEngine, query: str, k: int = 5
) -> List[Dict]:
    results = search_engine.search_dense(query)
    retrieved_results = []
    if results and hasattr(results, "points") and results.points:
        for point in results.points[:k]:
            payload = point.payload
            retrieved_results.append(
                {
                    "question": payload.get("question", "N/A"),
                    "answer": payload.get("answer", "N/A"),
                    "score": point.score,
                    "id": payload.get("id", "N/A"),
                }
            )
    return retrieved_results


def calculate_mrr(relevance_scores: List[int]) -> float:
    for i, score in enumerate(relevance_scores, 1):
        if score == 1:
            return 1 / i
    return 0


def calculate_ndcg(relevance_scores: List[int], k: int) -> float:
    dcg = sum(
        (2**rel - 1) / np.log2(i + 1) for i, rel in enumerate(relevance_scores[:k], 1)
    )
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = sum(
        (2**rel - 1) / np.log2(i + 1) for i, rel in enumerate(ideal_relevance[:k], 1)
    )
    return dcg / idcg if idcg > 0 else 0


def calculate_cosine_similarity(
    embedding1: np.ndarray, embedding2: np.ndarray
) -> float:
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )


def evaluate_model(
    model_name: str,
    ensemble_model: EnsembleModel,
    original_data: List[Dict],
    ground_truth: List[Dict],
    search_engine: SearchEngine,
    embeddings_model: EmbeddingsModel,
    batch_size: int = 100,
) -> List[Dict]:
    results = []

    # Pre-compute ground truth dictionary for faster lookup
    gt_dict = {gt["id"]: gt for gt in ground_truth}

    for i in tqdm(
        range(0, len(original_data), batch_size),
        desc=f"Evaluating {model_name}",
    ):
        batch = original_data[i : i + batch_size]
        batch_results = []

        # Perform dense search for all items in the batch
        questions = [item["question"] for item in batch]
        batch_retrieved_results = [
            perform_dense_search(search_engine, q) for q in questions
        ]

        # Generate LLM answers for all items in the batch
        prompts = []
        for item, retrieved in zip(batch, batch_retrieved_results):
            context = " ".join(
                [
                    f"Q: {r.get('question', '')} A: {r.get('answer', '')}"
                    for r in retrieved
                ]
            )
            prompt = f"Context: {context}\n\nUser: {item['question']}\nAssistant:"
            prompts.append(prompt)

        if model_name == "ensemble":
            batch_generated_responses = ensemble_model._generate_ensemble(prompts)
        else:
            batch_generated_responses = ensemble_model._generate_single_model(
                model_name, prompts
            )

        # Process each item in the batch
        for item, retrieved_results, answer_llm in zip(
            batch, batch_retrieved_results, batch_generated_responses
        ):
            original_id = item["id"]
            original_question = item["question"]
            original_answer = item["answer"]

            if original_id in gt_dict:
                # Generate embeddings
                original_embedding, _ = embeddings_model.embeddings([original_answer])
                llm_embedding, _ = embeddings_model.embeddings([answer_llm])

                # Calculate relevance scores
                relevance_scores = []
                for result in retrieved_results:
                    result_embedding, _ = embeddings_model.embeddings(
                        [result["answer"]]
                    )
                    similarity = calculate_cosine_similarity(
                        original_embedding[0], result_embedding[0]
                    )
                    relevance_scores.append(1 if similarity > 0.8 else 0)

                relevant_retrieved = sum(relevance_scores)
                precision = (
                    relevant_retrieved / len(retrieved_results)
                    if retrieved_results
                    else 0
                )
                recall = relevant_retrieved / 1  # Assuming 1 ground truth item per id
                mrr = calculate_mrr(relevance_scores)
                ndcg = calculate_ndcg(relevance_scores, k=5)
                cosine_sim = calculate_cosine_similarity(
                    original_embedding[0], llm_embedding[0]
                )

                batch_results.append(
                    {
                        "id": original_id,
                        "answer_llm": answer_llm,
                        "answer_orig": original_answer,
                        "question": original_question,
                        "precision": precision,
                        "recall": recall,
                        "mrr": mrr,
                        "ndcg": ndcg,
                        "cosine_similarity": cosine_sim,
                        "search_score": (
                            retrieved_results[0]["score"] if retrieved_results else None
                        ),
                    }
                )

        results.extend(batch_results)

    return results


def main():
    mlflow.set_experiment("RAG_Evaluation")
    with mlflow.start_run(run_name="model_comparison_evaluation"):
        original_data, ground_truth = load_datasets()
        ensemble_model = EnsembleModel()
        search_engine = SearchEngine(settings.COLLECTION_NAME_LOCAL)
        embeddings_model = EmbeddingsModel.get_instance()

        all_results = {}
        for model_name in tqdm(["gpt2", "dialogpt", "distilgpt2"], desc="Models"):
            print(f"Evaluating {model_name}...")
            model_results = evaluate_model(
                model_name,
                ensemble_model,
                original_data,
                ground_truth,
                search_engine,
                embeddings_model,
            )
            all_results[model_name] = model_results

            # Calculate and log average metrics
            avg_metrics = {
                "precision": np.mean([r["precision"] for r in model_results]),
                "recall": np.mean([r["recall"] for r in model_results]),
                "mrr": np.mean([r["mrr"] for r in model_results]),
                "ndcg": np.mean([r["ndcg"] for r in model_results]),
                "cosine_similarity": np.mean(
                    [r["cosine_similarity"] for r in model_results]
                ),
            }

            mlflow.log_metrics({f"{model_name}_{k}": v for k, v in avg_metrics.items()})

        # Print comparison results
        print("\nModel Comparison Results:")
        for model_name, results in all_results.items():
            avg_metrics = {
                "precision": np.mean([r["precision"] for r in results]),
                "recall": np.mean([r["recall"] for r in results]),
                "mrr": np.mean([r["mrr"] for r in results]),
                "ndcg": np.mean([r["ndcg"] for r in results]),
                "cosine_similarity": np.mean([r["cosine_similarity"] for r in results]),
            }
            print(f"\n{model_name.upper()}:")
            for metric, value in avg_metrics.items():
                print(f"{metric}: {value:.4f}")

        # Save detailed results to CSV
        all_results_flat = [
            {**result, "model": model_name}
            for model_name, results in all_results.items()
            for result in results
        ]
        df_results = pd.DataFrame(all_results_flat)
        df_results.to_csv("model_comparison_results.csv", index=False)
        print("\nDetailed results saved to 'model_comparison_results.csv'")

        # Log the results CSV as an artifact
        mlflow.log_artifact("model_comparison_results.csv")


if __name__ == "__main__":
    main()
