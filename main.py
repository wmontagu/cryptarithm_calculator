from fastapi import FastAPI, Request, Form
import joblib
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import time
import pandas as pd



from crypto import solve_crypto_init_medium, print_solution_medium
from hardversion import make_features, train_classifier #make a .py later...

app = FastAPI(title="Cryptarithm Calculator", description="Solving crptarithm puzzles using CSP and ML")


templates = Jinja2Templates(directory="templates")


os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

try:
    string = 'model.pkl'
    ml_model = joblib.load('model.pkl')
    MODEL = True
    print(f'Model {string} successfully uploaded!')
except Exception as e:
    print(f"Error: {e}")



def predict_solvability(str1, str2, equal, operation):

    if not MODEL:
        return None, "Model is not available"
    
    try:
        features_df = make_features(str1, str2, equal, operation) # This is a dataframe

        pred = ml_model.predict(features_df)[0]

        prob = ml_model.predict_proba(features_df)[0]

        conf = max(prob)

        return pred, f"Prediction: {'Solvable' if pred == 1 else 'Not Solvable'} {conf}"
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

@app.get('/', resposne_class=HTMLResponse)
async def read_root(request: Request):
    # Main page info
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "model_available": MODEL
    })


@app.post("/solve")
# Want to either do easy or hard...
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

    if not all(word.isalpha() for word in {str1, str2, result}):
        return templates.TemplateResponse('index.html', {
            "request": request,
            "error": "All inputs must only contain letters",
            "model_available": MODEL
        })
    
    all_chars = set(str1 + str2 + result)

    if len(all_chars) > 10:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Too many unique characters (need less than 10)",
            "model_available": MODEL
        })
    
    response_data = {"request": request,
                     "str1": str1,
                     "str2": str2,
                     "result": result,
                     "operation": operation,
                     "method": method,
                     "model_available": MODEL}
    
    if method == 'hard' and MODEL:
        pred, message = predict_solvability(str1, str2, result, operation)
        response_data["prediction"] = message


        if pred == 0:
            response_data['ml_unsolvable'] = True
    elif method == 'hard' and not MODEL:
        response_data["error"] = "ML method requires available model"
        method = "medium"
    
    return templates.TemplateResponse("index.html", response_data)


    


    


