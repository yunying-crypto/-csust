"""
Report generator - produces JSON, CSV, terminal, and HTML reports.
"""
import csv
import json
import logging
import os
import datetime

from jinja2 import Environment, FileSystemLoader

from models import EvaluationResult
from aggregator import compute_summary

logger = logging.getLogger(__name__)

# Jinja2 template directory
_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def generate_json_report(results, output_path):
    data = {
        "generated_at": datetime.datetime.now().isoformat(),
        "summary": compute_summary(results),
        "results": [r.to_dict() for r in results],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON report saved to {output_path}")
    return output_path


def generate_csv_report(results, output_path):
    if not results:
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            f.write("")
        return output_path
    fieldnames = [
        "problem_id", "domain", "question", "reference_answer",
        "intern_answer", "is_correct", "confidence",
        "judge_explanation", "error_type", "correct_answer_judge",
        "inference_tokens", "judge_tokens",
        "inference_latency", "judge_latency",
        "inference_error", "judge_error",
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "problem_id": r.problem_id,
                "domain": r.domain or "",
                "question": r.question,
                "reference_answer": r.reference_answer or "",
                "intern_answer": r.intern_answer,
                "is_correct": r.is_correct,
                "confidence": r.confidence,
                "judge_explanation": r.judge_explanation,
                "error_type": r.error_type or "",
                "correct_answer_judge": r.correct_answer_judge or "",
                "inference_tokens": r.inference_tokens,
                "judge_tokens": r.judge_tokens,
                "inference_latency": r.inference_latency,
                "judge_latency": r.judge_latency,
                "inference_error": r.inference_error or "",
                "judge_error": r.judge_error or "",
            })
    logger.info(f"CSV report saved to {output_path}")
    return output_path


def print_summary(results):
    summary = compute_summary(results)
    print("\n" + "=" * 60)
    print("  MATH AGENT EVALUATION REPORT")
    print("=" * 60)
    print(f"  Total Problems:    {summary['total']}")
    print(f"  Correct:           {summary['correct']}")
    print(f"  Accuracy:          {summary['accuracy']}%")
    print(f"  Avg Confidence:    {summary['avg_confidence']}")
    print(f"  Avg Inf. Latency:  {summary['avg_inference_latency']}s")
    print(f"  Avg Judge Latency: {summary['avg_judge_latency']}s")
    print(f"  Total Tokens:      {summary['total_inference_tokens'] + summary['total_judge_tokens']}")
    print("-" * 60)
    if summary["error_types"]:
        print("  Error Distribution:")
        for etype, count in summary["error_types"].items():
            print(f"    {etype}: {count}")
    if summary["domain_stats"]:
        print("-" * 60)
        print("  Domain Accuracy:")
        for domain, stats in summary["domain_stats"].items():
            print(f"    {domain}: {stats['accuracy']}% ({stats['correct']}/{stats['total']})")
    print("-" * 60)
    print("  Per-Problem Results:")
    for i, r in enumerate(results, 1):
        status = "PASS" if r.is_correct else "FAIL"
        print(f"  [{i}] {status} | {r.problem_id}: {r.intern_answer[:60]}")
    print("=" * 60)


def generate_html_report(results, output_path):
    """Generate HTML report using Jinja2 template."""
    summary = compute_summary(results)

    # Derive extra summary fields for template
    correct = summary["correct"]
    total = summary["total"]
    incorrect = sum(1 for r in results if not r.is_correct and not r.inference_error and not r.judge_error)
    error_count = sum(1 for r in results if r.inference_error or r.judge_error)
    total_inference_time = sum(
        r.inference_latency for r in results if not r.inference_error
    )
    total_tokens = summary["total_inference_tokens"] + summary["total_judge_tokens"]

    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=False)
    template = env.get_template("report.html")

    html = template.render(
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        summary={
            **summary,
            "incorrect": incorrect,
            "error": error_count,
            "total_inference_time": round(total_inference_time, 2),
            "total_tokens": total_tokens,
        },
        results=results,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML report saved to {output_path}")
    return output_path
