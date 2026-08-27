import json
from pathlib import Path

from churn_copilot.vector_store import search_policy_matches


CASES_PATH = (
    Path(__file__).resolve().parent
    / "retrieval_cases.json"
)


def evaluate_retrieval() -> None:
    with open(CASES_PATH, encoding="utf-8") as file:
        cases = json.load(file)

    correct = 0

    for case in cases:
        results = search_policy_matches(
            query=case["query"],
            top_k=1,
        )

        predicted_policy = results[0][0]["id"]
        score = results[0][1]

        expected_policy = case["expected_policy"]

        is_correct = predicted_policy == expected_policy

        if is_correct:
            correct += 1

        print(
            f"Query: {case['query']}\n"
            f"Expected: {expected_policy}\n"
            f"Predicted: {predicted_policy}\n"
            f"Score: {score:.3f}\n"
            f"Correct: {is_correct}\n"
        )

    accuracy = correct / len(cases)

    print(
        f"Top-1 retrieval accuracy: "
        f"{accuracy:.2%} "
        f"({correct}/{len(cases)})"
    )


if __name__ == "__main__":
    evaluate_retrieval()