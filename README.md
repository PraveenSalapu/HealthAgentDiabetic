# Diabetes Predictor & Health Assistant

This application provides diabetes risk prediction and an AI-powered health assistant.

## Prerequisites

1.  **Python 3.8+**
2.  **Install Dependencies:**
    ```bash
    pip install streamlit pandas numpy plotly google-generativeai joblib xgboost imbalanced-learn requests
    ```

## Environment Variables

Set the following environment variables for full functionality (especially for the Chatbot and Provider Search):

-   `GEMINI_API_KEY`: Your Google Gemini API key (Required for Chatbot)
-   `GOOGLE_PLACES_API_KEY`: (Optional) For real Google Places provider search
-   `USE_GOOGLE_PLACES`: Set to `true` to enable Google Places search
-   `ENABLE_GEMINI_BROWSER`: Set to `true` to enable Gemini Browser Use for booking

## Running the Application

There are two versions of the application:

### 1. Standard Version (`app.py`)
This is the base version with Diabetes Prediction and Chatbot.

```bash
streamlit run app.py
```

### 2. Advanced Version (`app2.py`)
This version includes **Provider Search** and **Appointment Scheduling** features.

```bash
streamlit run app2.py
```

## Troubleshooting

-   **Model Loading Errors:** Ensure the `model_output` (for `app.py`) or `model_output2` (for `app2.py`) directories contain the necessary `.pkl` and `.json` files.
-   **API Errors:** Verify your `GEMINI_API_KEY` is set correctly.
