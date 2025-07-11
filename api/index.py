# Complete app without the problematic components
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import os
import sys
import time
import importlib.util
from pathlib import Path

print("Starting complete app test...")

def import_module(module_name, file_path):
    """Dynamically import a module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import modules
current_dir = os.path.dirname(__file__)
crypto_module = import_module("crypto", os.path.join(current_dir, "crypto.py"))
inference_module = import_module("inference", os.path.join(current_dir, "inference.py"))

solve_crypto_init_medium = crypto_module.solve_crypto_init_medium
make_features = inference_module.make_features
load_sklearn_model = inference_module.load_sklearn_model

print("✅ All imports successful")

# Create FastAPI app
app = FastAPI(title="Cryptarithm Calculator", description="Solving cryptarithm puzzles using CSP and ML")

# Setup templates
BASE_DIR = Path("/var/task")
TEMPLATES_DIR = BASE_DIR / 'templates'
STATIC_DIR = BASE_DIR / 'static'
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

print(f"Templates directory: {TEMPLATES_DIR}")
print(f"✅ Templates initialized")

# Load model
MODEL = False
ml_model = None
try:
    model_path = os.path.join(current_dir, 'model.json')
    if os.path.exists(model_path):
        ml_model = load_sklearn_model(model_path)
        MODEL = ml_model is not None
        print(f"✅ Model loaded: {MODEL}")
except Exception as e:
    print(f"Model error: {e}")

# Use ML Model
def predict_solvability(str1, str2, equal, operation):
    if not MODEL or ml_model is None:
        return None, "Model is not available"
    
    try:
        features_df = make_features(str1, str2, equal, operation)
        pred = ml_model.predict([features_df])[0]
        prob = ml_model.predict_proba([features_df])[0]
        conf = max(prob)
        return pred, f"Prediction: {'Solvable' if pred == 1 else 'Not Solvable'} (confidence: {conf:.3f})"
    except Exception as e:
        return None, f"Prediction error: {e}"

def format_solution_output(assignment, str1, str2, result_str, operation):
    if not assignment:
        return None
    
    def word_to_number(word):
        return int(''.join(str(assignment[char]) for char in word))
    
    num1 = word_to_number(str1)
    num2 = word_to_number(str2)
    result = word_to_number(result_str)
    
    return {
        'assignments': {var: assignment[var] for var in sorted(assignment.keys())},
        'num1': num1,
        'num2': num2,
        'result': result,
        'equation': f"{num1} {operation} {num2} = {result}",
        'verification': f"Verification: {num1} {operation} {num2} = {result}"
    }


@app.get('/', response_class=HTMLResponse)
async def read_root(request: Request):
    # Extract query parameters if they exist
    str1 = request.query_params.get('str1', '')
    str2 = request.query_params.get('str2', '')
    result = request.query_params.get('result', '')
    operation = request.query_params.get('operation', '+')
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_available": MODEL,
        "str1": str1,
        "str2": str2,
        "result": result,
        "operation": operation
    })

@app.post("/solve", response_class=HTMLResponse)
async def solve_cryptarithm(
    request: Request,
    str1: str = Form(...),
    str2: str = Form(...),
    result: str = Form(...),
    operation: str = Form(...),
    method: str = Form(...)
):
    str1 = str1.strip().upper()
    str2 = str2.strip().upper()
    result = result.strip().upper()

    if not all(word.isalpha() for word in [str1, str2, result]):
        return templates.TemplateResponse('index.html', {
            "request": request,
            "error": "All inputs must only contain letters",
            "model_available": MODEL,
            "str1": str1,
            "str2": str2,
            "result": result,
            "operation": operation,
            "method": method
        })
    
    all_chars = set(str1 + str2 + result)
    if len(all_chars) > 10:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Too many unique characters (maximum 10 allowed for digits 0-9)",
            "model_available": MODEL,
            "str1": str1,
            "str2": str2,
            "result": result,
            "operation": operation,
            "method": method
        })
    
    response_data = {
        "request": request,
        "str1": str1,
        "str2": str2,
        "result": result,
        "operation": operation,
        "method": method,
        "model_available": MODEL
    }
    
    if method == 'hard' or method == 'both':
        if MODEL:
            pred, message = predict_solvability(str1, str2, result, operation)
            response_data["prediction"] = message
            response_data["prediction_available"] = True
            if pred == 0:
                response_data['ml_predicts_unsolvable'] = True
        else:
            response_data["prediction_available"] = False
            if method == 'hard':
                response_data["info"] = "ML model not available. Consider using Medium mode instead."
    
    if method == 'medium' or method == 'both':
        start_time = time.time()
        assignment = solve_crypto_init_medium(str1, str2, result, operation)
        solve_time = round(time.time() - start_time, 4)
        response_data["solve_time"] = solve_time
        
        if assignment:
            solution_data = format_solution_output(assignment, str1, str2, result, operation)
            response_data.update({
                "solved": True,
                "solution_data": solution_data
            })
        else:
            response_data["no_solution"] = True
    
    return templates.TemplateResponse("index.html", response_data)

@app.get("/examples", response_class=HTMLResponse)
async def get_examples(request: Request):
    examples = [
        {
            "str1": "SEND", "str2": "MORE", "result": "MONEY", "operation": "+", 
            "description": "",
            "difficulty": "hard"
        },
        {
            "str1": "TWO", "str2": "TWO", "result": "FOUR", "operation": "+", 
            "description": "",
            "difficulty": "easy"
        },
        {
            "str1": "GREAT", "str2": "LARGE", "result": "SMALL", "operation": "-", 
            "description": "",
            "difficulty": "hard"
        },
        {
            "str1": "US", "str2": "US", "result": "YEAH", "operation": "*", 
            "description": "",
            "difficulty": "medium"
        }
    ]
    
    return templates.TemplateResponse("examples.html", {
        "request": request,
        "examples": examples,
        "model_available": MODEL
    })

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {
        "request": request,
        "model_available": MODEL
    })

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ml_model_available": MODEL,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "current_directory": os.getcwd()
    }

print("✅ Complete FastAPI app created successfully")
