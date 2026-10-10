
from .router import SemanticRouter
from .examples import AGENT_EXAMPLES

def evaluate():
    router = SemanticRouter()

    total = 0
    correct = 0
    errors = []

    for expected_agent, examples in AGENT_EXAMPLES.items():
        for query in examples:
            predicted, confidence = router.classify(query)
            total += 1
            if predicted == expected_agent:
                correct += 1
            else:
                errors.append({
                    "query": query,
                    "expected": expected_agent,
                    "predicted": predicted,
                    "confidence": confidence,
                })

    accuracy = correct / total
    print(f"\n📊 Router Evaluation")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{total})")

    if errors:
        print(f"\n❌ {len(errors)} errors:")
        for e in errors:
            print(f"  '{e['query']}' → expected: {e['expected']}, got: {e['predicted']} ({e['confidence']:.2f})")

if __name__ == "__main__":
    evaluate()