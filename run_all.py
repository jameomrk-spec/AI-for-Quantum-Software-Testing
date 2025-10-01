import os

print("🚀 Starting full pipeline...")

# Step 1: Generate circuits
print("\n[1/4] Generating circuits...")
os.system("python src/generate_circuits.py")

# Step 2: Extract features
print("\n[2/4] Extracting features...")
os.system("python src/feature_extraction.py")

# Step 3: Train model
print("\n[3/4] Training model...")
os.system("python src/train_model.py")

# Step 4: Analyze results
print("\n[4/4] Analyzing results...")
os.system("python src/results_analysis.py")

print("\n✅ Pipeline complete! Check the 'results/' folder for figures and reports.")
