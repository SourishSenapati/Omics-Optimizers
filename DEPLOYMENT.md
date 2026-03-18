# Omics Optimizers Deployment Guide

This document provides the necessary instructions and configuration files for the complete online deployment of the **Omics Optimizers** platform.

## Architecture
- **Backend**: FastAPI running on Google Cloud Run.
- **Frontend**: Streamlit running on Streamlit Cloud (or Cloud Run).

---

## 1. Backend Deployment (Google Cloud Run)

The backend handles the PINN training, therapeutics queries (via PubChem), and the ProMED forensic feed.

### Prerequisites
- [Google Cloud SDK (gcloud)](https://cloud.google.com/sdk/docs/install) installed.
- A Google Cloud Project.

### Steps
1. **Build and Submit the Container**:
   Run the following in the project root:
   ```bash
   gcloud builds submit --tag gcr.io/[PROJECT_ID]/omics-backend backend/
   ```

2. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy omics-backend \
     --image gcr.io/[PROJECT_ID]/omics-backend \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 2
   ```
   *Note: Note down the URL returned after deployment (e.g., `https://omics-backend-xyz.a.run.app`).*

---

## 2. Frontend Deployment (Streamlit Cloud)

Streamlit Cloud is the easiest way to deploy the frontend dashboard.

### Steps
1. Push the code to your GitHub Repository: `https://github.com/SourishSenapati/Omics-Optimizers`.
2. Visit [share.streamlit.io](https://share.streamlit.io/).
3. Connect your repository.
4. **Configuration (CRITICAL)**:
   In the Streamlit app settings, add the following **Secret** or **Environment Variable**:
   ```toml
   BACKEND_URL = "https://omics-backend-xyz.a.run.app"
   ```
   *(Replace with your actual backend URL from Step 1).*

---

## 3. Automated Deployment (GitHub Actions)

We have created a workflow file `.github/workflows/deploy.yml` that automatically deploys the backend to Cloud Run on every push to `main`.

### Setup
1. Go to your GitHub Repository **Settings > Secrets and variables > Actions**.
2. Add the following secrets:
   - `GCP_PROJECT_ID`: Your Google Cloud Project ID.
   - `GCP_SA_KEY`: A JSON Key for a Service Account with Cloud Run and Cloud Build permissions.

---

## 4. Local Testing with Docker

To test the entire system locally using Docker Compose:

1. Create a `docker-compose.yml` (already provided in the repository).
2. Run:
   ```bash
   docker-compose up --build
   ```

---

*Prepared by Antigravity for Omics Optimizers | Team ORDINAR*
