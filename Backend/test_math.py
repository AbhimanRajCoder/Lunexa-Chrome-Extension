import math
import asyncio
from chat import calculate_consistency, calculate_cace, calculate_cars

def test_consistency():
    print("Testing Consistency...")
    # Case 1: Perfect consistency (variance 0)
    sims = [0.9, 0.9, 0.9]
    c = calculate_consistency(sims)
    print(f"Perfect Consistency (0.9, 0.9, 0.9): {c} (Expected ~1.0)")
    
    # Case 2: High variance
    sims = [0.1, 0.9, 0.5]
    c = calculate_consistency(sims)
    print(f"High Variance (0.1, 0.9, 0.5): {c}")
    
    # Case 3: Empty (Failed model)
    c = calculate_consistency([])
    print(f"Empty/Failed: {c} (Expected 0.05)")

def test_cace():
    print("\nTesting CACE...")
    # Case 1: Distances
    dists = [0.1, 0.2, 0.1]
    cace = calculate_cace(dists)
    print(f"CACE ([0.1, 0.2, 0.1]): {cace}")
    
    # Case 2: Empty
    cace = calculate_cace([])
    print(f"CACE Empty: {cace}")

def test_cars():
    print("\nTesting CARS...")
    metrics = {
        "factual_accuracy": 0.9,
        "reasoning_depth": 0.8,
        "source_verification": 0.7
    }
    consistency = 0.95
    cars = calculate_cars(metrics, consistency)
    print(f"CARS (High scores): {cars}")
    
    metrics_low = {
        "factual_accuracy": 0.05,
        "reasoning_depth": 0.05,
        "source_verification": 0.05
    }
    consistency_low = 0.05
    cars_low = calculate_cars(metrics_low, consistency_low)
    print(f"CARS (Low scores): {cars_low}")

if __name__ == "__main__":
    test_consistency()
    test_cace()
    test_cars()
