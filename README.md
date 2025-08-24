# Intelligent Resume Ranker 🤖

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E)
![spaCy](https://img.shields.io/badge/spaCy-3.7-09A3D5)

An interactive web application built with Streamlit that uses a machine learning model to automatically rank resumes based on their relevance to a given job description. This tool helps recruiters and hiring managers to quickly identify the most qualified candidates from a large pool of applicants, saving time and reducing manual effort.


## ✨ Features

-   **Automated Ranking:** Ranks multiple resumes against a job description using a predictive model.
-   **File Support:** Accepts resume uploads in both **PDF** and **TXT** formats.
-   **Interactive UI:** A simple and intuitive web interface built with Streamlit.
-   **NLP-Powered:** Utilizes Natural Language Processing (spaCy) for intelligent text preprocessing and feature extraction.
-   **Feature-Rich Model:** The ranking is not just based on keyword matching but on a model trained on features like cosine similarity, job category, and extracted skills, education, and experience sections.

---

## 🛠️ Technology Stack

| Technology      | Description                                                                 |
| --------------- | --------------------------------------------------------------------------- |
| **Python**      | The core programming language for the entire project.                       |
| **Streamlit**   | Used to build and serve the interactive web application UI.                 |
| **Scikit-learn**| For machine learning, including `TfidfVectorizer` and `LinearRegression`. |
| **spaCy**       | For advanced NLP tasks like lemmatization and stop-word removal.            |
| **Pandas**      | For data manipulation and displaying the final ranked results.              |
| **PyMuPDF**     | For robust and efficient text extraction from PDF files.                    |
| **Joblib**      | For serializing and loading the trained machine learning model and vectorizers. |

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

-   Python 3.10 or higher
-   Git for cloning the repository

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Joymangolch/Intelligent-Resume-Ranker.git
    cd Intelligent-Resume-Ranker
    ```

2.  **Create and activate a virtual environment:**
    This is a crucial step to keep project dependencies isolated.
    ```bash
    # Create the virtual environment
    python -m venv venv
    ```
    
    *Activate the environment:*
    -   **On Windows (PowerShell):**
        ```powershell
        .\venv\Scripts\activate
        ```
    -   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the spaCy language model:**
    The application uses a specific NLP model from spaCy that needs to be downloaded separately.
    ```bash
    python -m spacy download en_core_web_sm
    ```

### Running the Application

Once the installation is complete, you can launch the Streamlit application with a single command:

```bash
streamlit run app.py
```

Your default web browser will open a new tab with the application running at `http://localhost:8501`.

---

## 📁 Project Structure

The project is organized in a modular way to separate concerns:

```
resume_ranker_app/
├── artifacts/
│   ├── linear_regression_resume_model.joblib  # The trained ML model
│   ├── model_columns.joblib                   # Schema of the training data
│   ├── tfidf_vectorizer_cosine.joblib       # Vectorizer for similarity calculation
│   └── ... (other vectorizer and data files)
├── venv/
│   └── (Virtual environment files)
├── app.py                  # Main Streamlit application file (Frontend)
├── utils.py                # Helper functions for backend processing (NLP, feature prep)
├── requirements.txt        # List of all Python dependencies
└── README.md               # You are here!
```

---

## 🧠 How It Works: The Methodology

The application's ranking intelligence is based on a pre-trained Linear Regression model. Here's a high-level overview of the pipeline:

1.  **Text Preprocessing:** Both the job description and uploaded resumes undergo a rigorous cleaning process using spaCy (lemmatization, stop-word removal, etc.).
2.  **Feature Engineering:** A feature vector is constructed for each resume, which includes:
    -   A **cosine similarity score** between the resume and job description (calculated using TF-IDF).
    -   **One-hot encoded job category**.
    -   **Vectorized text** from heuristically extracted `skills`, `education`, and `experience` sections.
3.  **Prediction:** This complete feature vector is fed into the saved `LinearRegression` model, which predicts a final relevance score.
4.  **Ranking:** The application collects the scores for all uploaded resumes and displays them in a sorted, user-friendly table.

---

## 🔮 Future Enhancements

This project serves as a strong foundation. Future improvements could include:

-   [ ] **Advanced Information Extraction:** Replace regex with a custom-trained Named Entity Recognition (NER) model to more accurately identify skills, universities, job titles, etc.
-   [ ] **Semantic Understanding:** Integrate transformer-based models (like BERT or Sentence-Transformers) to understand the context and semantic meaning, not just keyword overlap.
-   [ ] **Explainable AI (XAI):** Enhance the UI to highlight which keywords or sections in a resume contributed most to its high score.
-   [ ] **Containerization:** Package the application using Docker for easier deployment to cloud services.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 👤 Author

**Joymangol**
-   GitHub: [(https://github.com/Joymangolch)]
-   LinkedIn: [(https://www.linkedin.com/in/joymangol-chingangbam-70b813288/)]

```
