"""
Federated Averaging (FedAvg) Simulator
Simulates cloud-based federated averaging process locally
This mimics what would happen if weights were aggregated in the cloud
"""

import os
import json
import time
from typing import List, Dict, Any


class FedAvgSimulator:
    """Simulates federated averaging process that would occur in the cloud"""
    
    def __init__(self, disease_folder: str):
        self.disease_folder = disease_folder
        self.hospitals = ["hospital_a", "hospital_b", "hospital_c"]
        self.weights_dir = os.path.join("data", "weights", disease_folder)
    
    def simulate_cloud_aggregation(self, simulate_delay: bool = True, delay_seconds: float = 0.5) -> Dict[str, Any]:
        """
        Simulates the cloud aggregation process:
        1. Fetch weights from each hospital (simulates network requests)
        2. Aggregate weights using FedAvg algorithm
        3. Return aggregated model
        
        Args:
            simulate_delay: Whether to simulate network delays
            delay_seconds: Time to delay each hospital fetch (simulating network latency)
        
        Returns:
            Aggregated weights dictionary
        """
        print("🌐 Simulating Cloud-Based Federated Averaging...")
        print(f"📡 Connecting to {len(self.hospitals)} hospitals...\n")
        
        # Step 1: Simulate fetching weights from each hospital
        hospital_weights = []
        for i, hospital in enumerate(self.hospitals, 1):
            print(f"  [{i}/{len(self.hospitals)}] Fetching weights from {hospital.replace('_', ' ').title()}...")
            
            if simulate_delay:
                time.sleep(delay_seconds)  # Simulate network latency
            
            weights_path = os.path.join(self.weights_dir, f"{hospital}_weights.json")
            
            if os.path.exists(weights_path):
                with open(weights_path, "r") as f:
                    weights = json.load(f)
                    hospital_weights.append(weights)
                    print(f"       ✓ Received {len(weights['features'])} features")
            else:
                print(f"       ✗ Hospital weights not found: {weights_path}")
        
        if not hospital_weights:
            print("\n❌ No hospital weights found!")
            return None
        
        # Step 2: Simulate cloud aggregation process
        print("\n☁️  Cloud Server: Aggregating weights using FedAvg algorithm...")
        time.sleep(delay_seconds * 0.5)  # Simulate cloud processing time
        
        # Perform federated averaging
        aggregated = self._fedavg_algorithm(hospital_weights)
        
        print(f"✓ Aggregation complete: {len(hospital_weights)} hospitals averaged")
        print(f"✓ Final model: {len(aggregated['features'])} features, {len(aggregated['coef'][0])} coefficients")
        print()
        
        return aggregated
    
    def _fedavg_algorithm(self, hospital_weights: List[Dict]) -> Dict[str, Any]:
        """
        Federated Averaging (FedAvg) Algorithm
        This is the standard federated learning aggregation algorithm
        
        Formula: w_global = Σ(n_k * w_k) / Σ(n_k)
        For simplicity, we use equal weighted average: w_global = (1/K) * Σ(w_k)
        
        Args:
            hospital_weights: List of weight dictionaries from each hospital
            
        Returns:
            Aggregated weights dictionary
        """
        if not hospital_weights:
            return None
        
        # Get feature list from first hospital (should be same for all)
        features = hospital_weights[0]["features"]
        num_features = len(features)
        num_hospitals = len(hospital_weights)
        
        # Initialize aggregated weights
        coef_sum = [0.0] * num_features
        intercept_sum = 0.0
        
        # Aggregate coefficients and intercepts
        for weights in hospital_weights:
            coef = weights["coef"][0]  # Logistic regression has single coefficient array
            intercept = weights["intercept"][0]
            
            # Sum all weights
            for i in range(num_features):
                coef_sum[i] += coef[i]
            intercept_sum += intercept
        
        # Average: Simple average (equal weight for each hospital)
        # In real FedAvg, you might weight by number of samples: avg = sum(n_k * w_k) / sum(n_k)
        avg_coef = [c / num_hospitals for c in coef_sum]
        avg_intercept = intercept_sum / num_hospitals
        
        # Return aggregated model
        return {
            "model": "federated_logistic_regression",
            "aggregation_method": "fedavg",
            "num_hospitals": num_hospitals,
            "features": features,
            "coef": [avg_coef],
            "intercept": [avg_intercept],
            "classes": hospital_weights[0]["classes"],
            "metadata": {
                "simulated": True,
                "timestamp": time.time(),
                "hospitals_aggregated": [w["hospital"] for w in hospital_weights]
            }
        }
    
    def get_weighted_fedavg(self, sample_counts: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Perform weighted federated averaging based on sample counts
        Formula: w_global = Σ(n_k * w_k) / Σ(n_k)
        
        Args:
            sample_counts: Dictionary mapping hospital names to their sample counts
                          If None, uses equal weights
        
        Returns:
            Weighted aggregated weights
        """
        # Load hospital weights
        hospital_weights = []
        for hospital in self.hospitals:
            weights_path = os.path.join(self.weights_dir, f"{hospital}_weights.json")
            if os.path.exists(weights_path):
                with open(weights_path, "r") as f:
                    weights = json.load(f)
                    hospital_weights.append(weights)
        
        if not hospital_weights:
            return None
        
        # If no sample counts provided, use equal weights
        if sample_counts is None:
            return self._fedavg_algorithm(hospital_weights)
        
        # Weighted FedAvg
        features = hospital_weights[0]["features"]
        num_features = len(features)
        
        total_samples = 0
        weighted_coef_sum = [0.0] * num_features
        weighted_intercept_sum = 0.0
        
        for weights in hospital_weights:
            hospital_name = weights["hospital"].lower().replace(" ", "_")
            n_k = sample_counts.get(hospital_name, 1)  # Default to 1 if not specified
            total_samples += n_k
            
            coef = weights["coef"][0]
            intercept = weights["intercept"][0]
            
            # Weight by sample count
            for i in range(num_features):
                weighted_coef_sum[i] += n_k * coef[i]
            weighted_intercept_sum += n_k * intercept
        
        # Normalize by total samples
        avg_coef = [c / total_samples for c in weighted_coef_sum]
        avg_intercept = weighted_intercept_sum / total_samples
        
        return {
            "model": "federated_logistic_regression",
            "aggregation_method": "weighted_fedavg",
            "num_hospitals": len(hospital_weights),
            "total_samples": total_samples,
            "features": features,
            "coef": [avg_coef],
            "intercept": [avg_intercept],
            "classes": hospital_weights[0]["classes"],
            "metadata": {
                "simulated": True,
                "weighted": True,
                "sample_counts": sample_counts
            }
        }


def simulate_federated_averaging(disease_folder: str, simulate_delay: bool = False) -> Dict[str, Any]:
    """
    Convenience function to simulate federated averaging
    
    Args:
        disease_folder: Disease folder name (e.g., "diabetes", "asthma")
        simulate_delay: Whether to simulate network delays
    
    Returns:
        Aggregated weights dictionary
    """
    simulator = FedAvgSimulator(disease_folder)
    return simulator.simulate_cloud_aggregation(simulate_delay=simulate_delay)


