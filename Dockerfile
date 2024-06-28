# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Copy necessary files
COPY . .

# Install dependencies
# RUN pip install pandas scikit-learn joblib

# Make run_workflow.sh executable
RUN chmod +x run_workflow.sh

# Run the workflow
CMD ["./run_workflow.sh"]
