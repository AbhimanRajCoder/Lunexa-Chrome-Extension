import sys
import os

# Add current directory to sys.path to import chat
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chat import compute_CARS, calculate_structure_score

def test_structure_score():
    print("Testing calculate_structure_score...")
    
    # Case 1: Short garbage
    short = "idk"
    s1 = calculate_structure_score(short)
    print(f"Short text score: {s1} (Expected ~0.1)")
    assert s1 == 0.1, f"Expected 0.1, got {s1}"

    # Case 2: Good sentence
    good = "The capital of France is Paris. It is a beautiful city."
    s2 = calculate_structure_score(good)
    print(f"Good text score: {s2} (Expected > 0.5)")
    assert s2 > 0.5, f"Expected > 0.5, got {s2}"

    # Case 3: Long formatted text
    long_text = "This is a longer answer.\nIt has multiple lines.\nAnd proper punctuation!"
    s3 = calculate_structure_score(long_text)
    print(f"Long formatted text score: {s3} (Expected >= 0.7)")
    assert s3 >= 0.7, f"Expected >= 0.7, got {s3}"
    
    print("Structure score tests passed!\n")

def test_cars_threshold():
    print("Testing compute_CARS threshold...")
    
    # Case 1: Just below old threshold (0.25) but above new (0.15)
    metrics_survivor = {
        "factualAccuracy_judge": 0.2,
        "factualAccuracy_grounded": 0.16, # > 0.15, should survive
        "reasoningDepth": 0.5,
        "external_confidence": 0.5,
        "consistency": 0.5,
        "sourceVerification": 0.5,
        "structure_score": 0.8
    }
    answer = "Some answer"
    score = compute_CARS(metrics_survivor, answer)
    print(f"Survivor score: {score} (Expected > 0)")
    assert score > 0, f"Model should not be eliminated! Score: {score}"

    # Case 2: Below new threshold
    metrics_eliminated = {
        "factualAccuracy_judge": 0.2,
        "factualAccuracy_grounded": 0.10, # < 0.15, should die
        "reasoningDepth": 0.5,
        "external_confidence": 0.5,
        "consistency": 0.5,
        "sourceVerification": 0.5,
        "structure_score": 0.8
    }
    score_dead = compute_CARS(metrics_eliminated, answer)
    print(f"Eliminated score: {score_dead} (Expected -1.0)")
    assert score_dead == -1.0, f"Model should be eliminated! Score: {score_dead}"
    
    print("Threshold tests passed!\n")

if __name__ == "__main__":
    try:
        test_structure_score()
        test_cars_threshold()
        print("ALL TESTS PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
