from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
import voyageai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Static files and templates setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the VoyageAI API key
api_key = os.getenv("VOYAGEAI_API_KEY")
if not api_key:
    raise ValueError("API key is not found. Please make sure to set the VOYAGEAI_API_KEY in your .env file.")

# Load the trained SVM model
model_file = 'svm_classifier_model.pkl'
try:
    with open(model_file, 'rb') as file:
        loaded_model = pickle.load(file)
except Exception as e:
    raise ValueError(f"Failed to load the model: {str(e)}")

# Initialize VoyageAI client
try:
    vo = voyageai.Client(api_key=api_key)
except Exception as e:
    raise ValueError(f"Failed to initialize VoyageAI client: {str(e)}")


def predict(text: str):
    """
    Predict the class of input text using the trained model.
    """
    try:
        # Generate embedding for input text
        embedding_response = vo.embed([text], model="voyage-3", input_type="document")
        if not embedding_response.embeddings:
            raise ValueError("Failed to generate embeddings.")

        embedding = embedding_response.embeddings[0]

        # Make prediction
        prediction = loaded_model.predict([embedding])[0]
        confidence = loaded_model.predict_proba([embedding]).max()

        return {"prediction": prediction, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the home page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/classify_experience", response_class=HTMLResponse)
async def classify(request: Request, input_text: str = Form(...)):
    """
    Handle the classification request.
    """
    try:
        result = predict(input_text)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": result,
                "input_text": input_text
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": str(e),
                "input_text": input_text
            },
        )

