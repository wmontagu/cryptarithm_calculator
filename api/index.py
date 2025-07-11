from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import sys
import time
import pickle
import json
from pathlib import Path
from crypto import solve_crypto_init_medium
from inference import make_features, load_sklearn_model

# For Pydantic v2 compatibility
os.environ["PYDANTIC_VALIDATION_ERROR_SERIALIZE_JSON_NAR"] = "1"

app = FastAPI(title="Cryptarithm Calculator", description="Solving cryptarithm puzzles using CSP and ML")

# Add startup debugging information
print(f"Starting application in directory: {os.getcwd()}")
print(f"__file__: {__file__}")
print(f"Script directory: {os.path.dirname(__file__)}")

# Determine if we're running on Vercel or locally
IS_VERCEL = os.environ.get('VERCEL', '0') == '1'

# Set up paths - critical for both local and Vercel deployment
BASE_DIR = Path(os.path.dirname(os.path.dirname(__file__))) if os.path.dirname(__file__) else Path.cwd()
API_DIR = Path(os.path.dirname(__file__)) if os.path.dirname(__file__) else Path.cwd()
print(f"BASE_DIR: {BASE_DIR}")
print(f"API_DIR: {API_DIR}")

# In Vercel, the root directory is the deployment directory
# In local development, we need to account for running from api/ subdirectory
TEMPLATES_DIR = API_DIR / 'templates'
STATIC_DIR = BASE_DIR / 'static'
print(f"TEMPLATES_DIR: {TEMPLATES_DIR}")

# Check if files and directories exist before app starts
try:
    print(f"Checking content of current directory: {os.listdir('.')}")
except Exception as e:
    print(f"Error listing current directory: {str(e)}")

# Ensure directories exist
try:
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    print(f"Created templates directory at {TEMPLATES_DIR}")
    print(f"Created static directory at {STATIC_DIR}")
except Exception as e:
    print(f"Error creating directories: {str(e)}")

# Mount static files
try:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    print(f"Static files mounted from {STATIC_DIR}")
except Exception as e:
    print(f"Error mounting static files: {str(e)}")

# Initialize templates
templates = None
try:
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    print(f"Templates directory set to: {TEMPLATES_DIR}")
    if os.path.exists(TEMPLATES_DIR / 'index.html'):
        print("✅ index.html template found")
    else:
        print("❌ index.html template NOT found!")
    print(f"Templates initialized from {TEMPLATES_DIR}")
    # Check if template files exist
    try:
        template_files = os.listdir(TEMPLATES_DIR)
        print(f"Available template files: {template_files}")
    except Exception as e:
        print(f"Error listing template files: {str(e)}")
except Exception as e:
    print(f"Error initializing templates: {str(e)}")

MODEL = False
ml_model = None
try:
    # Try multiple paths for model loading in both local and Vercel environments
    possible_paths = [
        os.path.join(API_DIR, 'model.json'),  # First check in api directory
        os.path.join(BASE_DIR, 'model.json'),  # Then check in root directory
        'model.json'  # Fallback to current directory
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found model at: {path}")
            model_path = path
            break
        else:
            print(f"Model not found at: {path}")
    
    if model_path:
        ml_model = load_sklearn_model(model_path)
        MODEL = True
        print(f'Model successfully loaded from: {model_path}')
    else:
        print(f"Model file not found in any of these paths: {possible_paths}")
        
except Exception as e:
    print(f"ML Model loading error: {str(e)}")
    import traceback
    print(traceback.format_exc())
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
    try:
        if templates is None:
            return HTMLResponse(content="<h1>Error: Templates not available</h1>", status_code=500)
        
        try:
            # Check if index.html exists
            template_path = os.path.join(TEMPLATES_DIR, "index.html")
            if not os.path.exists(template_path):
                print(f"Template file not found: {template_path}")
                return HTMLResponse(content=f"<h1>Error: Template not found: {template_path}</h1>", status_code=500)
                
            return templates.TemplateResponse("index.html", {
                "request": request, 
                "model_available": MODEL
            })
        except Exception as e:
            print(f"Template rendering error: {str(e)}")
            return HTMLResponse(content=f"<h1>Template rendering error: {str(e)}</h1>", status_code=500)
    except Exception as e:
        print(f"Error in read_root: {str(e)}")
        return HTMLResponse(content=f"<h1>Server error: {str(e)}</h1>", status_code=500)

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

@app.get('/favicon.ico', include_in_schema=False)
async def get_favicon():
    favicon_path = os.path.join(STATIC_DIR, 'favicon.ico')
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return HTMLResponse(status_code=204)  # No content response if favicon doesn't exist

@app.get("/health")
async def health_check():
    # Get environment details for debugging
    env_info = {}
    for key in ["VERCEL", "VERCEL_ENV", "VERCEL_URL", "VERCEL_REGION", "PYTHON_VERSION"]:
        if key in os.environ:
            env_info[key] = os.environ[key]
            
    # Check directories and files
    try:
        file_list = os.listdir('.')
    except Exception as e:
        file_list = f"Error listing files: {str(e)}"
        
    try:
        template_exists = os.path.exists(TEMPLATES_DIR)
        template_files = os.listdir(TEMPLATES_DIR) if template_exists else []
    except Exception as e:
        template_files = f"Error listing template files: {str(e)}"
    
    return {
        "status": "healthy",
        "ml_model_available": MODEL,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "current_directory": os.getcwd(),
        "environment_info": env_info,
        "files_in_root": file_list,
        "template_dir_exists": template_exists if 'template_exists' in locals() else None,
        "template_files": template_files
    }

# For Vercel, we expose the app directly
# The __name__ == "__main__" block is kept for local development

# Export app variable directly for Vercel serverless
# Vercel Python uses ASGI without requiring mangum
__all__ = ['app']

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Cryptarithm Calculator...")
    print(f"📊 ML Model Available: {MODEL}")
    print("🌐 Access the app at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)