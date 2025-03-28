import evaluate

def calculate_metrics(predictions, references):
    # Load the metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    
    print(predictions)
    print(references)
    # Compute BLEU scores
    bleu1 = bleu_metric.compute(predictions=predictions, references=references, max_order=1)['bleu']
    bleu2 = bleu_metric.compute(predictions=predictions, references=references, max_order=2)['bleu']
    
    # Compute ROUGE-L
    rouge_l = rouge_metric.compute(predictions=predictions, references=references)['rougeL']
    
    # Compute METEOR
    meteor_score = meteor_metric.compute(predictions=predictions, references=references)['meteor']
    
    # Return as a dictionary
    return {
        "bleu1": bleu1,
        "bleu2": bleu2,
        "rougeL": rouge_l,
        "meteor": meteor_score
    }
