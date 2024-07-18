from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(ground_truth, predictions):
    precision = precision_score(ground_truth, predictions, average='weighted')
    recall = recall_score(ground_truth, predictions, average='weighted')
    f1 = f1_score(ground_truth, predictions, average='weighted')
    return precision, recall, f1

if __name__ == "__main__":
    ground_truth = [...]  # Load ground truth data
    predictions = [...]  # Get predictions from your model
    precision, recall, f1 = evaluate(ground_truth, predictions)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
