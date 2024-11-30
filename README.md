# Text Classification API

## Overview

This is a FastAPI-based text classification application that uses VoyageAI embeddings and an SVM classifier to predict
the class of input text.

## Prerequisites

- Python 3.8+
- FastAPI
- VoyageAI
- scikit-learn
- python-dotenv

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mustafoyev-202/Classification_fastapi.git
cd classification
```

### 2. Install Dependencies

```bash
pip install fastapi uvicorn voyageai python-dotenv Jinja2 scikit-learn python-multipart
```

### 3. Environment Configuration

Create a `.env` file in the project root and add your VoyageAI API key:

```
VOYAGEAI_API_KEY=your_voyageai_api_key_here
```

### 4. Run the Application

```bash
uvicorn main:app --host 127.0.0.1 --port 5000
```

## API Endpoints

### 1. Home Page

- **URL:** `http://127.0.0.1:5000/`
- **Method:** GET
- **Description:** Renders the home page with the text classification interface

### 2. Text Classification

- **URL:** `http://127.0.0.1:5000/classify_experience`
- **Method:** POST
- **Content-Type:** `application/x-www-form-urlencoded`

#### Request Parameters

| Parameter    | Type   | Required | Description               |
|--------------|--------|----------|---------------------------|
| `input_text` | string | Yes      | The text to be classified |

#### Response

The API returns a JSON-like response with two key fields:

- `prediction`: The predicted class of the text
- `confidence`: The confidence score of the prediction (between 0 and 1)

### Example Request

You can test the API using curl:

```bash
curl -X POST http://127.0.0.1:5000/classify_experience \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "input_text=Your text to classify here"
```

### Web Interface

Access the classification interface by navigating to `http://127.0.0.1:5000/` in your web browser.

## Troubleshooting

- Ensure your VoyageAI API key is correctly set in the `.env` file
- Verify all dependencies are installed
- Check that the SVM model file `svm_classifier_model.pkl` is present in the project directory

## Notes

- The model uses VoyageAI's `voyage-3` embedding model
- Classification is performed using a pre-trained SVM classifier
- Errors will be displayed if embedding or prediction fails