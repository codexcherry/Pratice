Alright Sir 👍
Here is **P1** again — **shortest**, **no extra**, **directly write in record**.

---

# **P1: Git – User Registration Form**

## **Aim**

To create a user registration form and perform basic Git operations.

---

## **index.html**

```html
<!DOCTYPE html>
<html>
<body>
<h2>Event Registration</h2>

<form>
Name:<br>
<input type="text"><br><br>

Email:<br>
<input type="email"><br><br>

Phone:<br>
<input type="text"><br><br>

<button>Register</button>
</form>
</body>
</html>
```

---

## **Git Commands**

```bash
git init
git add index.html
git commit -m "Initial form"
```

---

## **Create Branch & Update**

```bash
git checkout -b update-form
```

Add:

```html
Department:<br>
<input type="text"><br><br>
```

```bash
git add index.html
git commit -m "Added department field"
```

---

## **Merge & Push**

```bash
git checkout main
git merge update-form
git push origin main
```

---

## **Result**

User registration form created, updated using a new branch, merged into main, and pushed to GitHub.

---

Say **“Next”** when ready 👍






--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Alright Sir 👍
Here is **P2**, **very simple**, **no extra**, **directly lab-record ready**.

---

# **P2: Continuous Integration using Jenkins**

## **Aim**

To configure Jenkins to automatically build a project whenever code is pushed to GitHub.

---

## **Step 1: Install Java (Required for Jenkins)**

```bash
sudo apt update
sudo apt install openjdk-17-jdk -y
```

Check:

```bash
java -version
```

---

## **Step 2: Install Jenkins**

```bash
sudo apt install jenkins -y
```

Start Jenkins:

```bash
sudo systemctl start jenkins
sudo systemctl enable jenkins
```

---

## **Step 3: Open Jenkins**

Open browser:

```
http://localhost:8080
```

Get admin password:

```bash
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
```

✔ Install **Suggested Plugins**
✔ Create admin user

---

## **Step 4: Create Freestyle Job**

1. Click **New Item**
2. Job name: `student-portal`
3. Select **Freestyle project**
4. Click **OK**

---

## **Step 5: Configure GitHub Repository**

Under **Source Code Management**:

* Select **Git**
* Repository URL:

```
https://github.com/your-username/student-portal.git
```

---

## **Step 6: Enable Auto Build**

Under **Build Triggers**:
✔ Select **GitHub hook trigger for GITScm polling**

Save the job.

---

## **Step 7: Push Change to GitHub**

Edit HTML file:

```html
<h3>Updated Page</h3>
```

Push:

```bash
git add index.html
git commit -m "Updated HTML page"
git push origin main
```

---

## **Step 8: Verify Jenkins Build**

* Jenkins automatically triggers a build
* Build status shows **SUCCESS**

---

## **Result**

Jenkins Freestyle job was configured to pull code from GitHub and automatically trigger a build whenever changes were pushed.

---

Say **“p3”** when ready 👍



-------------------------------------------------------------------------------------


Alright Sir 👍
Here is **P3**, **short**, **simple**, **Ubuntu-based**, **lab-record ready**.

---

# **P3: Docker – Build and Run Web Application**

## **Aim**

To create a simple web application, build a Docker image, run a container, and execute basic Docker commands.

---

## **Step 1: Install Docker**

```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
```

Check:

```bash
docker --version
```

---

## **Step 2: Create Project Folder**

```bash
mkdir p3
cd p3
```

---

## **Step 3: Create HTML File**

**index.html**

```html
<!DOCTYPE html>
<html>
<body>
<h2>Hello Docker</h2>
</body>
</html>
```

---

## **Step 4: Create Dockerfile**

```dockerfile
FROM nginx
COPY index.html /usr/share/nginx/html/index.html
```

---

## **Step 5: Build Docker Image**

```bash
docker build -t webapp .
```

---

## **Step 6: Run Docker Container**

```bash
docker run -d -p 8080:80 --name webcontainer webapp
```

Open browser:

```
http://localhost:8080
```

---

## **Step 7: Docker Commands**

```bash
docker ps
docker images
docker stop webcontainer
docker rm webcontainer
docker rmi webapp
```

---

## **Result**

Docker image was built and run successfully, and the application was accessed through the browser.

---

Say **“p4”** when ready 👍



-------------------------------------------------------------------------------------
Alright Sir 👍
Here is **P4**, **short**, **clear**, **Ubuntu-based**, **lab-record ready**.

---

# **P4: Deploy Docker Application on Kubernetes**

## **Aim**

To deploy a Docker application on Kubernetes, expose it using NodePort, and scale the deployment.

---

## **Step 1: Install Minikube & kubectl**

```bash
sudo apt update
sudo apt install -y curl kubectl
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

Start Minikube:

```bash
minikube start
```

---

## **Step 2: Create Deployment File**

**deployment.yaml**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-deploy
spec:
  replicas: 1
  selector:
    matchLabels:
      app: webapp
  template:
    metadata:
      labels:
        app: webapp
    spec:
      containers:
      - name: web
        image: webapp
        ports:
        - containerPort: 80
```

Apply:

```bash
kubectl apply -f deployment.yaml
```

---

## **Step 3: Verify Pod**

```bash
kubectl get pods
```

---

## **Step 4: Create Service**

**service.yaml**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-service
spec:
  type: NodePort
  selector:
    app: webapp
  ports:
    - port: 80
      targetPort: 80
      nodePort: 30007
```

Apply:

```bash
kubectl apply -f service.yaml
```

---

## **Step 5: Access Application**

```bash
minikube ip
```

Open:

```
http://<minikube-ip>:30007
```

---

## **Step 6: Scale Deployment**

```bash
kubectl scale deployment web-deploy --replicas=3
```

Verify:

```bash
kubectl get pods
```

---

## **Result**

Docker application was deployed on Kubernetes, exposed using NodePort, and scaled to multiple replicas.

---

Say **“p5”** when ready 👍
--------------------------------------------------------------------------------------------

Alright Sir 👍
Here is **P5**, **short**, **simple**, **Ubuntu-based**, **lab-record ready**.

---

# **P5: ML Environment Setup**

## **Aim**

To set up a simple ML environment and verify installation using Jupyter Notebook.

---

## **Step 1: Install Python Tools**

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```

---

## **Step 2: Create Project Folder**

```bash
mkdir p5
cd p5
```

---

## **Step 3: Create Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## **Step 4: Create `requirements.txt`**

```text
numpy
pandas
scikit-learn
matplotlib
jupyter
```

---

## **Step 5: Install Packages**

```bash
pip install -r requirements.txt
```

---

## **Step 6: Verify Using Jupyter**

```bash
jupyter notebook
```

Create `test.ipynb` and run:

```python
import numpy, pandas, sklearn, matplotlib
print("ML setup successful")
```

---

## **Step 7: Commit to Git**

```bash
git init
git add requirements.txt test.ipynb
git commit -m "ML environment setup"
git push origin main
```

---

## **Result**

ML environment was successfully set up and verified using Jupyter Notebook.

---

Say **“p6”** when ready 👍


------------------------------------------------------------------------------------------------------

Alright Sir 👍
Here is **P6**, **short**, **simple**, **Ubuntu-based**, **lab-record ready**.

---

# **P6: Multi-Container Application using Docker Compose**

## **Aim**

To design and deploy a multi-container application using Docker Compose with an application service and a Redis service.

---

## **Step 1: Install Docker & Docker Compose**

```bash
sudo apt update
sudo apt install docker.io docker-compose -y
sudo systemctl start docker
sudo systemctl enable docker
```

---

## **Step 2: Create Project Folder**

```bash
mkdir p6
cd p6
```

---

## **Step 3: Create Application File**

**app.py**

```python
from flask import Flask
import redis

app = Flask(__name__)
r = redis.Redis(host='redis', port=6379)

@app.route('/')
def home():
    r.incr('count')
    return f"Visitor Count: {r.get('count').decode()}"

app.run(host='0.0.0.0', port=5000)
```

---

## **Step 4: Create Dockerfile**

```dockerfile
FROM python:3.9
WORKDIR /app
COPY app.py .
RUN pip install flask redis
CMD ["python", "app.py"]
```

---

## **Step 5: Create docker-compose.yml**

```yaml
version: "3"
services:
  app:
    build: .
    ports:
      - "5000:5000"
    depends_on:
      - redis

  redis:
    image: redis
```

---

## **Step 6: Run Application**

```bash
docker-compose up
```

Open browser:

```
http://localhost:5000
```

---

## **Step 7: Stop Application**

```bash
docker-compose down
```

---

## **Result**

A multi-container application was created using Docker Compose and inter-container communication was verified.

---

Say **“p7”** when ready 👍
------------------------------------------------------------------------------------------------

Alright Sir 👍
Here is **P7**, **short**, **simple**, **Ubuntu-based**, **lab-record ready**.

---

# **P7: Data Version Control (DVC)**

## **Aim**

To track raw and processed data using DVC and create a reproducible data pipeline.

---

## **Step 1: Install DVC**

```bash
pip install dvc
```

---

## **Step 2: Create Project Folder**

```bash
mkdir p7
cd p7
git init
dvc init
```

---

## **Step 3: Create Raw Data**

```bash
mkdir data
nano data/raw.txt
```

Add:

```
10
20
30
```

---

## **Step 4: Track Data**

```bash
dvc add data/raw.txt
git add data/raw.txt.dvc .gitignore
git commit -m "Track raw data"
```

---

## **Step 5: Create Processing Script**

**process.py**

```python
with open("data/raw.txt") as f:
    nums = [int(x) for x in f]

with open("data/processed.txt", "w") as f:
    for n in nums:
        f.write(str(n*2) + "\n")
```

---

## **Step 6: Create Pipeline**

```bash
dvc run -n process_data \
-d process.py -d data/raw.txt \
-o data/processed.txt \
python process.py
```

---

## **Step 7: Reproduce**

```bash
dvc repro
```

---

## **Result**

Data was tracked using DVC and a reproducible data pipeline was created successfully.

---

Say **“p8”** when ready 👍

----------------------------------------------------------------------------------------------------

Alright Sir 👍
Here is **P8**, **short**, **simple**, **Ubuntu-based**, **lab-record ready**.

---

# **P8: Experiment Tracking using MLflow**

## **Aim**

To track machine learning experiments using MLflow by logging parameters and metrics.

---

## **Step 1: Install MLflow**

```bash
pip install mlflow scikit-learn
```

---

## **Step 2: Create Python File**

**mlflow_demo.py**

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)

    print("Accuracy:", acc)
```

---

## **Step 3: Start MLflow UI**

```bash
mlflow ui
```

Open browser:

```
http://localhost:5000
```

---

## **Step 4: Run Experiment**

Open **new terminal**:

```bash
python mlflow_demo.py
```

Run multiple times to create multiple experiments.

---

## **Result**

MLflow successfully tracked model parameters and accuracy, and experiments were compared using MLflow UI.

---

Say **“p9”** when ready 👍


---------------------------------------------------------------------------------------------------

Alright Sir 👍
Here is **P9**, **short**, **simple**, **Ubuntu-based**, **lab-record ready**.

---

# **P9: Model Optimization & Inference using ONNX**

## **Aim**

To export a trained machine learning model to ONNX format and run inference using ONNX Runtime.

---

## **Step 1: Install Required Packages**

```bash
pip install scikit-learn onnx onnxruntime skl2onnx
```

---

## **Step 2: Create Python File**

**onnx_demo.py**

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np

X, y = load_iris(return_X_y=True)

model = LogisticRegression(max_iter=200)
model.fit(X, y)

initial_type = [('input', FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
pred = session.run(None, {input_name: X[:1].astype(np.float32)})

print("ONNX prediction successful")
```

---

## **Step 3: Run Program**

```bash
python onnx_demo.py
```

---

## **Result**

The trained model was converted to ONNX format and inference was successfully performed using ONNX Runtime.

---

Say **“p10”** when ready 👍


---------------------------------------------------------------------------------------------------

Alright Sir 👍
Here is **P10**, **short**, **simple**, **Ubuntu-based**, **lab-record ready**.

---

# **P10: Serve ML Model using FastAPI**

## **Aim**

To serve a machine learning model using FastAPI with a `/predict` REST endpoint and test predictions.

---

## **Step 1: Install Required Packages**

```bash
pip install fastapi uvicorn scikit-learn pydantic
```

---

## **Step 2: Create FastAPI Application**

**app.py**

```python
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

app = FastAPI()

# Train model
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)
model.fit(X, y)

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}
```

---

## **Step 3: Run FastAPI Server**

```bash
uvicorn app:app --reload
```

Open browser:

```
http://127.0.0.1:8000/docs
```

---

## **Step 4: Test Prediction**

Input:

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

✔ Prediction returned successfully

---

## **Step 5: Create Dockerfile**

```dockerfile
FROM python:3.9
WORKDIR /app
COPY app.py .
RUN pip install fastapi uvicorn scikit-learn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## **Step 6: Build & Run Container**

```bash
docker build -t fastapi-ml .
docker run -d -p 8000:8000 fastapi-ml
```

Open:

```
http://localhost:8000/docs
```

---

## **Result**

A machine learning model was served using FastAPI with a `/predict` endpoint, tested successfully, and containerized using Docker.

---

✅ **All Programs P1–P10 Completed**
If you want, I can now:

* Convert everything into a **single lab manual**
* Prepare **viva questions & answers**
* Verify **screenshots / outputs**

Just tell me 👍

