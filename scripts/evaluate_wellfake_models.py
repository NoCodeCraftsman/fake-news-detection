import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import json
import numpy as np
from datetime import datetime
import logging
import re
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("L:/Important/MCA/Mini Project/fake_news_detection")

def load_test_dataset():
    """Load the preprocessed test dataset."""
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at: {test_path}")

    logger.info(f"Loading test dataset: {test_path}")
    df = pd.read_csv(test_path)
    df["text"] = df["text"].astype(str).fillna("")
    df["label"] = df["label"].astype(int)

    logger.info(f"Test dataset loaded: {len(df)} samples")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df

def compute_metrics(pred):
    """Compute evaluation metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def find_best_checkpoint(trial_dir):
    """Find the best checkpoint in a trial directory"""
    steps = []
    for sub in trial_dir.iterdir():
        m = re.match(r"checkpoint-(\d+)$", sub.name)
        if m and sub.is_dir():
            steps.append((int(m.group(1)), sub))
    return max(steps, key=lambda x: x[0])[1] if steps else None

def find_all_model_checkpoints():
    """Find all available model checkpoints and organize by model type"""
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return {}

    model_checkpoints = {
        "bert-base-uncased": [],
        "roberta-base": []
    }

    for trial_dir in sorted(results_dir.glob("trial_*")):
        logger.info(f"Scanning {trial_dir}")

        checkpoint = find_best_checkpoint(trial_dir)
        if not checkpoint:
            logger.warning(f"No checkpoint found in {trial_dir}")
            continue

        # Check config to determine model type
        config_file = checkpoint / "config.json"
        if not config_file.exists():
            logger.warning(f"No config.json found in {checkpoint}")
            continue

        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                model_name = config_data.get("_name_or_path", "")

            if "bert" in model_name.lower() and "roberta" not in model_name.lower():
                model_checkpoints["bert-base-uncased"].append({
                    "path": checkpoint,
                    "trial": trial_dir.name,
                    "config": config_data
                })
                logger.info(f"Found BERT checkpoint: {checkpoint}")

            elif "roberta" in model_name.lower():
                model_checkpoints["roberta-base"].append({
                    "path": checkpoint,
                    "trial": trial_dir.name,
                    "config": config_data
                })
                logger.info(f"Found RoBERTa checkpoint: {checkpoint}")
            else:
                logger.warning(f"Unknown model type in {checkpoint}: {model_name}")

        except Exception as e:
            logger.error(f"Error reading config from {config_file}: {e}")
            continue

    for model_type, checkpoints in model_checkpoints.items():
        logger.info(f"Found {len(checkpoints)} checkpoints for {model_type}")

    return model_checkpoints

def select_best_checkpoint(model_type, checkpoints):
    """Select the best checkpoint for a model type (for now, just use the first one)"""
    if not checkpoints:
        return None


    selected = checkpoints[0]
    logger.info(f"Selected checkpoint for {model_type}: {selected['path']} (from {selected['trial']})")
    return selected

def plot_confusion_matrix(cm, labels, model_name, results_folder):
    """Generate and save confusion matrix plot"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title(f"Confusion Matrix - {model_name}", fontsize=16, pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    plot_path = results_folder / f"{model_name.replace('/', '-')}_confusion_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix plot saved: {plot_path}")
    return str(plot_path)

def evaluate_model(model_name, checkpoint_info, test_texts, test_labels, results_folder):
    """Evaluate a single model"""
    if not checkpoint_info:
        logger.warning(f"No checkpoint available for {model_name}")
        return None

    checkpoint_path = checkpoint_info["path"]
    trial_name = checkpoint_info["trial"]

    logger.info(f"Evaluating {model_name} from {checkpoint_path} ({trial_name})")

    try:
        logger.info(f"Loading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

        def tokenize_fn(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

        test_ds = Dataset.from_dict({"text": test_texts, "label": test_labels}).map(tokenize_fn, batched=True)

        trainer = Trainer(model=model, compute_metrics=compute_metrics)

        logger.info(f"Running evaluation for {model_name}")
        eval_results = trainer.evaluate(test_ds)

        logger.info(f"Getting predictions for {model_name}")
        predictions = trainer.predict(test_ds)

        y_pred = predictions.predictions.argmax(-1)
        y_true = predictions.label_ids
        cm = confusion_matrix(y_true, y_pred)

        # Generate classification report
        clf_report = classification_report(y_true, y_pred, output_dict=True)

        # Generate confusion matrix plot
        plot_path = plot_confusion_matrix(cm, ["Real", "Fake"], model_name, results_folder)

        # Prepare detailed results
        detailed_results = {
            "model": model_name,
            "checkpoint_path": str(checkpoint_path),
            "trial": trial_name,
            "eval_results": eval_results,
            "classification_report": clf_report,
            "confusion_matrix": cm.tolist(),
            "sample_count": len(test_texts),
            "accuracy": float(eval_results.get("eval_accuracy", 0)),
            "f1_score": float(eval_results.get("eval_f1", 0)),
            "precision": float(eval_results.get("eval_precision", 0)),
            "recall": float(eval_results.get("eval_recall", 0)),
            "plot_path": plot_path
        }

        model_file = results_folder / f"{model_name.replace('/', '-')}_evaluation_results.json"
        with open(model_file, 'w') as f:
            json.dump(detailed_results, f, indent=4)

        logger.info(f"✅ {model_name} - Accuracy: {detailed_results['accuracy']:.4f}, F1: {detailed_results['f1_score']:.4f}")
        logger.info(f"   Results saved to: {model_file}")

        return detailed_results

    except Exception as e:
        logger.error(f"❌ Error evaluating {model_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def create_summary_report(all_results, results_folder, df_test):
    """Create comprehensive summary report"""
    timestamp = datetime.now().isoformat()

    summary = {
        "evaluation_timestamp": timestamp,
        "dataset_info": {
            "test_samples": len(df_test),
            "label_distribution": df_test["label"].value_counts().to_dict(),
            "average_text_length": float(df_test["text"].str.len().mean()),
            "dataset_source": "WELFake_preprocessed_test_set"
        },
        "models_evaluated": [],
        "results": {},
        "comparison": {},
        "summary_statistics": {}
    }

    accuracies = []
    f1_scores = []

    for model_name, results in all_results.items():
        if results:
            summary["models_evaluated"].append(model_name)
            summary["results"][model_name] = {
                "accuracy": results["accuracy"],
                "f1_score": results["f1_score"],
                "precision": results["precision"],
                "recall": results["recall"],
                "confusion_matrix": results["confusion_matrix"],
                "checkpoint_path": results["checkpoint_path"],
                "trial": results["trial"]
            }

            accuracies.append(results["accuracy"])
            f1_scores.append(results["f1_score"])

    if accuracies:
        summary["summary_statistics"] = {
            "models_count": len(all_results),
            "successful_evaluations": len([r for r in all_results.values() if r is not None]),
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
            "max_accuracy": float(max(accuracies)),
            "max_f1": float(max(f1_scores)),
            "min_accuracy": float(min(accuracies)),
            "min_f1": float(min(f1_scores))
        }

        best_acc_model = max(summary["results"].keys(), key=lambda x: summary["results"][x]["accuracy"])
        best_f1_model = max(summary["results"].keys(), key=lambda x: summary["results"][x]["f1_score"])

        summary["comparison"] = {
            "best_accuracy_model": best_acc_model,
            "best_accuracy_score": summary["results"][best_acc_model]["accuracy"],
            "best_f1_model": best_f1_model,
            "best_f1_score": summary["results"][best_f1_model]["f1_score"],
            "accuracy_difference": max(accuracies) - min(accuracies),
            "f1_difference": max(f1_scores) - min(f1_scores)
        }

    summary_file = results_folder / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)

    logger.info(f"📊 Summary report saved: {summary_file}")
    return summary

def create_comparison_plot(all_results, results_folder):
    """Create a comparison plot of model performances"""
    if len(all_results) < 2:
        logger.warning("Need at least 2 models to create comparison plot")
        return

    models = []
    accuracies = []
    f1_scores = []

    for model_name, results in all_results.items():
        if results:
            models.append(model_name.replace('-base-uncased', '').upper())
            accuracies.append(results["accuracy"])
            f1_scores.append(results["f1_score"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(models, accuracies, color=['#1f77b4', '#ff7f0e'])
    ax1.set_title('Model Accuracy Comparison', fontsize=14, pad=15)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1)

    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11)

    bars2 = ax2.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e'])
    ax2.set_title('Model F1 Score Comparison', fontsize=14, pad=15)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_ylim(0, 1)

    for bar, f1 in zip(bars2, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()

    comparison_plot_path = results_folder / "model_comparison.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Comparison plot saved: {comparison_plot_path}")
    return str(comparison_plot_path)

def main():
    """Main function to evaluate both BERT and RoBERTa models"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = PROJECT_ROOT / "logs" / f"evaluation_results_{timestamp}_wellfake_both_models"
    results_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"🚀 Starting evaluation of BERT and RoBERTa models")
    logger.info(f"Results will be saved to: {results_folder}")

    try:
        df_test = load_test_dataset()
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return

    test_texts = df_test["text"].tolist()
    test_labels = df_test["label"].tolist()

    logger.info("\n" + "="*50)
    logger.info("DISCOVERING MODEL CHECKPOINTS")
    logger.info("="*50)

    all_checkpoints = find_all_model_checkpoints()

    models_to_evaluate = ["bert-base-uncased", "roberta-base"]
    all_results = {}

    logger.info("\n" + "="*50)
    logger.info("EVALUATING MODELS")
    logger.info("="*50)

    for model_name in models_to_evaluate:
        logger.info(f"\n--- Processing {model_name} ---")

        checkpoints = all_checkpoints.get(model_name, [])
        if not checkpoints:
            logger.warning(f"❌ No checkpoints found for {model_name}")
            all_results[model_name] = None
            continue

        selected_checkpoint = select_best_checkpoint(model_name, checkpoints)

        result = evaluate_model(
            model_name, 
            selected_checkpoint, 
            test_texts, 
            test_labels, 
            results_folder
        )

        all_results[model_name] = result

    logger.info("\n" + "="*50)
    logger.info("CREATING SUMMARY REPORT")
    logger.info("="*50)

    summary = create_summary_report(all_results, results_folder, df_test)

    # Create comparison plot
    create_comparison_plot(all_results, results_folder)

    logger.info("\n🎉 EVALUATION COMPLETE!")
    logger.info(f"📁 All results saved in: {results_folder}")

    print("\n" + "="*70)
    print("FINAL EVALUATION RESULTS SUMMARY")
    print("="*70)
    print(f"Results folder: {results_folder}")
    print(f"Test samples: {len(df_test)}")
    print(f"Models evaluated: {len([r for r in all_results.values() if r is not None])}")

    if summary.get("summary_statistics"):
        stats = summary["summary_statistics"]
        print(f"\nOverall Performance:")
        print(f"Mean accuracy: {stats['mean_accuracy']:.4f} (±{stats['std_accuracy']:.4f})")
        print(f"Mean F1 score: {stats['mean_f1']:.4f} (±{stats['std_f1']:.4f})")

    if summary.get("comparison"):
        comp = summary["comparison"]
        print(f"\nBest Performers:")
        print(f"Best accuracy: {comp['best_accuracy_model']} ({comp['best_accuracy_score']:.4f})")
        print(f"Best F1 score: {comp['best_f1_model']} ({comp['best_f1_score']:.4f})")

    print(f"\nDetailed Results:")
    for model_name, result in all_results.items():
        if result:
            print(f"{model_name}:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  F1 Score: {result['f1_score']:.4f}")
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  Recall: {result['recall']:.4f}")
            print(f"  Checkpoint: {result['trial']}")
        else:
            print(f"{model_name}: ❌ Evaluation failed")

    print("="*70)

if __name__ == "__main__":
    main()
