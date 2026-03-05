# Heal.ai API

Heal.ai is an AI-powered clinical co-pilot designed to assist medical professionals by providing evidence-based answers from patient data and medical literature.

This API features PDF uploads for patient histories and maintains conversation memory for each patient.

---

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* Python 3.10+
* An active ANTHROPIC API Key

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/heal-ai-api.git](https://github.com/your-username/heal-ai-api.git)
    cd heal-ai-api
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Your Environment File:**
    Create a `.env` file in the project root and add your Google API key.
    ```bash
    echo 'ANTHROPIC_API_KEY="your-anthropic-api-key-goes-here"' > .env
    ```

5.  **Add Your Data:**
    Place your data source (e.g., `Gold_Standard.zip`) in the root of the project directory.

6.  **Build the Vector Database:**
    Run the build script to process your data and create the local FAISS database.
    ```bash
    python build_db.py
    ```

---

## ▶️ Running the Application

To run the FastAPI server, use the following command:

```bash
python -m uvicorn main:app --reload
