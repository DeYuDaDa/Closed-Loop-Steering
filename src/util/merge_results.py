import json
import os
import sys
import argparse
import numpy as np

def merge_results(file_paths, output_path):
    merged_data = {}

    for path in file_paths:
        if not os.path.exists(path):
            print(f"Warning: File {path} not found. Skipping.")
            continue
        
        print(f"Loading {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding {path}: {e}")
                continue

        for mode, stats in data.items():
            if mode not in merged_data:
                merged_data[mode] = stats
            else:
                print(f"Merging mode '{mode}' from {path}...")
                existing_stats = merged_data[mode]
                
                # Merge per_problem
                existing_problems = existing_stats.get('per_problem', [])
                new_problems = stats.get('per_problem', [])
                
                # Use a dict to avoid duplicate IDs if necessary, 
                # but usually we just want to combine them or replace
                # For this task, we assume they might be different problems or same problems with better results.
                # To be safe, if IDs overlap, we'll keep the one with 'correct': True or just the new one.
                combined_problems_dict = {p['id']: p for p in existing_problems}
                for p in new_problems:
                    pid = p['id']
                    if pid in combined_problems_dict:
                        # If overlap, prefer the correct one
                        if p.get('correct') or not combined_problems_dict[pid].get('correct'):
                            combined_problems_dict[pid] = p
                    else:
                        combined_problems_dict[pid] = p
                
                merged_problems = list(combined_problems_dict.values())
                existing_stats['per_problem'] = merged_problems
                
                # Recalculate metrics
                total_count = len(merged_problems)
                correct_count = sum(1 for p in merged_problems if p.get('correct'))
                accuracy = correct_count / total_count if total_count > 0 else 0
                
                existing_stats['total_count'] = total_count
                existing_stats['correct_count'] = correct_count
                existing_stats['accuracy'] = accuracy
                
                # Average other metrics if they exist
                for metric in ['repetition', 'ppl', 'tokens', 'local_dtr']:
                    vals = [p.get(metric) for p in merged_problems if p.get(metric) is not None]
                    if vals:
                        existing_stats[metric] = float(np.mean(vals))
                
                # For Dynamic_Spherical, update module_diagnostics if it exists
                if mode == "Dynamic_Spherical" and 'per_problem' in existing_stats:
                    problems = existing_stats['per_problem']
                    active_steps = [p.get('alpha_active_steps', 0) for p in problems]
                    total_steps = [p.get('num_tokens', 1) for p in problems]
                    max_alphas = [p.get('alpha_max_value', 0.0) for p in problems]
                    convergences = [p.get('convergence', False) for p in problems]
                    
                    existing_stats['module_diagnostics'] = {
                        "problems_with_intervention": sum(1 for s in active_steps if s > 0),
                        "problems_with_convergence": sum(1 for c in convergences if c),
                        "mean_alpha_active_ratio": float(np.mean([a/max(t, 1) for a, t in zip(active_steps, total_steps)])),
                        "mean_max_alpha": float(np.mean(max_alphas))
                    }

    print(f"Saving merged results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge experiment result JSON files.")
    parser.add_argument("files", nargs="+", help="JSON files to merge")
    parser.add_argument("--output", "-o", default="merged_results.json", help="Output JSON file path")
    
    args = parser.parse_args()
    
    if len(args.files) < 2:
        print("Error: At least two files are required to merge.")
        sys.exit(1)
        
    merge_results(args.files, args.output)
