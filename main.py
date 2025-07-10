from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import time
import pickle

from crypto import solve_crypto_init_medium, print_solution_medium
from inference import make_features

app = FastAPI(title="Cryptarithm Calculator", description="Solving cryptarithm puzzles using CSP and ML")


os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

MODEL = False
ml_model = None
try:
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            ml_model = pickle.load(f)
        MODEL = True
    print(f'Model model.pkl successfully uploaded!')
except Exception as e:
    print(f"ML Model not found: {e}")
    print("Running without ML model - only CSP method available")

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
    
    solution_data = {
        'assignments': {var: assignment[var] for var in sorted(assignment.keys())},
        'num1': num1,
        'num2': num2,
        'result': result,
        'equation': f"{num1} {operation} {num2} = {result}",
        'verification': f"Verification: {num1} {operation} {num2} = {result}"
    }
    return solution_data

@app.get('/', response_class=HTMLResponse)
async def read_root(request: Request):
    """Main page"""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "model_available": MODEL
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
            "description": "The classic cryptarithm puzzle",
            "difficulty": "hard"
        },
        {
            "str1": "TWO", "str2": "TWO", "result": "FOUR", "operation": "+", 
            "description": "Simple addition with repeated words",
            "difficulty": "easy"
        },
        {
            "str1": "ABC", "str2": "DEF", "result": "GHIJ", "operation": "+", 
            "description": "Pattern-based example",
            "difficulty": "medium"
        },
        {
            "str1": "CAB", "str2": "CAB", "result": "DEED", "operation": "+", 
            "description": "Repeated patterns with symmetry",
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
        "ml_model_available": MODEL
    }

# For Vercel, we expose the app directly
# The __name__ == "__main__" block is kept for local development

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Cryptarithm Calculator...")
    print(f"📊 ML Model Available: {MODEL}")
    print("🌐 Access the app at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)