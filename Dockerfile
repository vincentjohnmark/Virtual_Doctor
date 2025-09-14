# 1. Start with a specific, lightweight Python version as our base
FROM python:3.11.9-slim

# 2. Set the working directory inside the container to /app
# All subsequent commands will run from this directory
WORKDIR /app

# 3. Copy only the requirements file first to leverage Docker's caching
COPY requirements.txt .

# 4. Install all the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project files into the container's /app directory
# This single command copies main.py and all necessary folders like
# dataset/, model/, templates/, and static/
COPY . .

# 6. Expose the port the app will run on
EXPOSE 10000

# 7. Define the command to run your application using Gunicorn
# This is the command that starts your web server
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "main:app"]