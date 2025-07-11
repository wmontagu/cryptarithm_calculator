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

import importlib.util
import sys
import os
import json

def import_module(module_name, file_path):
    """Dynamically import a module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Usage
current_dir = os.path.dirname(__file__)
crypto_module = import_module("crypto", os.path.join(current_dir, "crypto.py"))
inference_module = import_module("inference", os.path.join(current_dir, "inference.py"))

# Access functions - updated with your actual function names
solve_crypto_init_medium = crypto_module.solve_crypto_init_medium
make_features = inference_module.make_features
load_sklearn_model = inference_module.load_sklearn_model

def handler(event, context):
    """Main serverless function handler"""
    try:
        # Example usage of your imported functions
        # Adjust parameters based on your actual function signatures
        
        # Use crypto function
        crypto_result = solve_crypto_init_medium()  # Add parameters as needed
        
        # Use inference functions
        features = make_features()  # Add parameters as needed
        model = load_sklearn_model()  # Add parameters as needed
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': True,
                'crypto_result': crypto_result,
                'features': features,
                'message': 'Functions executed successfully'
            })
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }

# Optional: Add debug information
print(f"Successfully imported:")
print(f"  - solve_crypto_init_medium from crypto")
print(f"  - make_features from inference") 
print(f"  - load_sklearn_model from inference")# In index.py
import importlib.util
import sys
import os
import json

def import_module(module_name, file_path):
    """Dynamically import a module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Usage
current_dir = os.path.dirname(__file__)
crypto_module = import_module("crypto", os.path.join(current_dir, "crypto.py"))
inference_module = import_module("inference", os.path.join(current_dir, "inference.py"))

# Access functions - updated with your actual function names
solve_crypto_init_medium = crypto_module.solve_crypto_init_medium
make_features = inference_module.make_features
load_sklearn_model = inference_module.load_sklearn_model

def handler(event, context):
    """Main serverless function handler"""
    try:
        # Example usage of your imported functions
        # Adjust parameters based on your actual function signatures
        
        # Use crypto function
        crypto_result = solve_crypto_init_medium()  # Add parameters as needed
        
        # Use inference functions
        features = make_features()  # Add parameters as needed
        model = load_sklearn_model()  # Add parameters as needed
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': True,
                'crypto_result': crypto_result,
                'features': features,
                'message': 'Functions executed successfully'
            })
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }

# Optional: Add debug information
print(f"Successfully imported:")
print(f"  - solve_crypto_init_medium from crypto")
print(f"  - make_features from inference") 
print(f"  - load_sklearn_model from inference")


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
except Exception as e:
    print(f"Error creating directories: {str(e)}")

# Static files directory setup
STATIC_DIR = os.path.join(os.getcwd(), 'static')

# Mount static files - with conditional handling for Vercel's read-only filesystem
try:
    # Only attempt to mount static files if the directory exists
    if os.path.exists(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
        print(f"Static files mounted from {STATIC_DIR}")
    else:
        print(f"Static directory not found at {STATIC_DIR} - static files will not be available")
        # In production Vercel environment, static files should be handled by Vercel's CDN
        if os.environ.get('VERCEL') == '1':
            print("Running on Vercel - static assets should be served by Vercel CDN")
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
        if ml_model is not None:
            MODEL = True
            print(f'Model successfully loaded from: {model_path}')
        else:
            MODEL = False
            print(f'Failed to load model from: {model_path}')
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

# For Vercel deployment we need both WSGI and ASGI compatibility
from http.server import BaseHTTPRequestHandler

# Create a BaseHTTPRequestHandler class for WSGI compatibility
class handler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'FastAPI app is running. Access via proper ASGI server.')
        return

# Export both the app and handler for Vercel
__all__ = ['app', 'handler']

# Note: The handler class above is for Vercel's WSGI compatibility
# The app itself is an ASGI application that will be used when possible

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Cryptarithm Calculator...")
    print(f"📊 ML Model Available: {MODEL}")
    print("🌐 Access the app at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)