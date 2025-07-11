from extract_random_forest import extract_sklearn_random_forest
import os

print("Starting model conversion process...")

# Convert the model from pickle to JSON
result = extract_sklearn_random_forest(
    model_path='model.pkl',  # Input pickle model
    output_path='model.json'  # Output JSON - naming it to match your app's expectations
)

if os.path.exists('model.json'):
    print(f"✅ Conversion successful! model.json created ({os.path.getsize('model.json')/1024:.2f} KB)")
    print("Your app should now be able to load this model without sklearn dependency")
else:
    print("❌ Conversion failed. Check errors above.")
