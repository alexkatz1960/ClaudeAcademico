FROM python:3.11-slim 
WORKDIR /app 
COPY requirements_interfaces.txt . 
RUN pip install --no-cache-dir -r requirements_interfaces.txt 
COPY interfaces ./interfaces 
CMD uvicorn interfaces.fastapi_backend.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run interfaces/streamlit_dashboard/dashboard.py --server.port 8501 --server.enableCORS false
