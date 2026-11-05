# app.py
"""
Sleep Apnea Detection System
- Unified Web Interface
- Sensor + ML (real-time)
- ML-only Prediction
- Time-window ML
- Alert logic + Sound alarm
"""

import os, re, time, json, threading, sqlite3
from datetime import datetime
from functools import wraps

from flask import (
    Flask, request, redirect, url_for,
    render_template_string, Response, session, flash, g
)
from werkzeug.security import generate_password_hash, check_password_hash

import numpy as np
import joblib

# ---------------- CONFIG ----------------
DATABASE = "users.db"
SECRET_KEY = "sleep-apnea-secret"

SERIAL_PORT = "COM3"      # change if needed
SERIAL_BAUD = 115200
USE_SERIAL = True

MODEL_PATH = "apnea_model.pkl"

HISTORY_SIZE = 60
SSE_INTERVAL = 0.8

# --- ML & ALERT PARAMETERS ---
WINDOW_SIZE = 15
ALERT_THRESHOLD = 80      # %
ALERT_COUNT_LIMIT = 5
# ----------------------------

try:
    import serial
except:
    serial = None

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY

# ---------------- LOAD ML MODEL ----------------
ml_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# ---------------- SHARED STATE ----------------
latest = {
    "bpm": 0.0,
    "spo2": 0.0,
    "bpm_hist": [],
    "spo2_hist": [],
    "time_hist": [],
    "status": "WAITING",
    "probability": None
}

lock = threading.Lock()
alert_counter = 0
CURRENT_MODE = "SENSOR"

# ---------------- DATABASE ----------------
def get_db():
    if "_db" not in g:
        g._db = sqlite3.connect(DATABASE)
        g._db.row_factory = sqlite3.Row
    return g._db

@app.teardown_appcontext
def close_db(e):
    db = g.pop("_db", None)
    if db:
        db.close()

def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT
        )
    """)
    db.commit()

# ---------------- AUTH ----------------
def login_required(f):
    @wraps(f)
    def wrap(*a, **k):
        if "user_id" not in session:
            return redirect("/login")
        return f(*a, **k)
    return wrap

# ---------------- ML ----------------
def ml_predict(bpm, spo2):
    X = np.array([[bpm, spo2]])
    pred = ml_model.predict(X)[0]
    prob = ml_model.predict_proba(X)[0][1] * 100
    return ("APNEA" if pred == 1 else "NORMAL"), round(prob, 2)

# ---------------- SERIAL ----------------
bpm_re = re.compile(r"BPM[:=]?\s*([\d.]+)", re.I)
spo2_re = re.compile(r"SPO2[:=]?\s*([\d.]+)", re.I)

def parse_line(line):
    bpm = spo2 = None
    m = bpm_re.search(line)
    if m:
        bpm = float(m.group(1))
    m = spo2_re.search(line)
    if m:
        spo2 = float(m.group(1))
    return bpm, spo2

# ---------------- SENSOR WORKER ----------------
def sensor_worker():
    global latest, alert_counter

    if not USE_SERIAL or serial is None:
        return

    try:
        s = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        print("[SERIAL] Connected")
    except:
        print("[SERIAL] Not connected")
        return

    while True:
        try:
            line = s.readline().decode(errors="ignore")
            bpm, spo2 = parse_line(line)

            if bpm is not None and spo2 is not None and CURRENT_MODE == "SENSOR":
                with lock:
                    latest["bpm"] = round(bpm, 2)
                    latest["spo2"] = round(spo2, 2)

                    latest["bpm_hist"].append(latest["bpm"])
                    latest["spo2_hist"].append(latest["spo2"])
                    latest["time_hist"].append(datetime.now().strftime("%H:%M:%S"))

                    for k in ("bpm_hist", "spo2_hist", "time_hist"):
                        latest[k] = latest[k][-HISTORY_SIZE:]

                    if len(latest["bpm_hist"]) >= WINDOW_SIZE:
                        bpm_avg = sum(latest["bpm_hist"][-WINDOW_SIZE:]) / WINDOW_SIZE
                        spo2_avg = sum(latest["spo2_hist"][-WINDOW_SIZE:]) / WINDOW_SIZE

                        status, prob = ml_predict(bpm_avg, spo2_avg)
                        latest["probability"] = prob

                        if prob >= ALERT_THRESHOLD:
                            alert_counter += 1
                        else:
                            alert_counter = 0

                        if alert_counter >= ALERT_COUNT_LIMIT:
                            latest["status"] = "APNEA ALERT"
                        else:
                            latest["status"] = status
                    else:
                        latest["status"] = "WAITING"
                        latest["probability"] = None
        except:
            pass

        time.sleep(0.2)

# ---------------- SSE ----------------
def event_stream():
    while True:
        with lock:
            yield f"data: {json.dumps(latest)}\n\n"
        time.sleep(SSE_INTERVAL)

# ---------------- MODE SWITCH ----------------
@app.route("/switch_mode")
@login_required
def switch_mode():
    global CURRENT_MODE
    CURRENT_MODE = "ML_ONLY" if CURRENT_MODE == "SENSOR" else "SENSOR"
    return redirect(request.referrer or "/monitor")

# ---------------- UI ----------------
base_head = """
<!DOCTYPE html>
<html>
<head>
<title>Sleep Apnea Detection</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body { background:#f6f7fb; }
</style>
</head>
<body>

<nav class="navbar navbar-expand-lg navbar-dark bg-primary">
<div class="container-fluid">
<span class="navbar-brand">Sleep Apnea Detection</span>
<div>
<a href="/switch_mode" class="btn btn-warning me-2">Switch Mode</a>
<a href="/monitor" class="btn btn-outline-light me-2">Live Monitor</a>
<a href="/predict" class="btn btn-outline-light me-2">Predict</a>
<a href="/logout" class="btn btn-light">Logout</a>
</div>
</div>
</nav>
"""

# ---------------- LOGIN ----------------
login_html = """
<!DOCTYPE html>
<html>
<head>
<title>Login</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body style="background:#f6f7fb">

<nav class="navbar navbar-dark bg-primary">
<div class="container-fluid">
<span class="navbar-brand">Sleep Apnea Detection</span>
</div>
</nav>

<div class="card shadow-sm mx-auto" style="max-width:420px;margin-top:80px">
<div class="card-header text-center bg-dark text-white">User Login</div>
<div class="card-body">
{% if get_flashed_messages() %}
<div class="alert alert-danger">{{ get_flashed_messages()[0] }}</div>
{% endif %}
<form method="post">
<input name="username" class="form-control mb-3" placeholder="Username" required>
<input type="password" name="password" class="form-control mb-3" placeholder="Password" required>
<button class="btn btn-primary w-100">Login</button>
</form>
</div>
<div class="card-footer text-center">
<a href="/register">Create account</a>
</div>
</div>

</body>
</html>
"""

# ---------------- REGISTER ----------------
register_html = """
<!DOCTYPE html>
<html>
<head>
<title>Register</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body style="background:#f6f7fb">

<nav class="navbar navbar-dark bg-primary">
<div class="container-fluid">
<span class="navbar-brand">Sleep Apnea Detection</span>
</div>
</nav>

<div class="card shadow-sm mx-auto" style="max-width:420px;margin-top:80px">
<div class="card-header text-center bg-success text-white">Register</div>
<div class="card-body">
{% if get_flashed_messages() %}
<div class="alert alert-danger">{{ get_flashed_messages()[0] }}</div>
{% endif %}
<form method="post">
<input name="username" class="form-control mb-3" required>
<input type="password" name="password" class="form-control mb-3" required>
<button class="btn btn-success w-100">Register</button>
</form>
</div>
</div>

</body>
</html>
"""

# ---------------- MONITOR ----------------
monitor_html = base_head + """
<div class="container mt-4">
<h2 class="text-center">Live Monitor</h2>
<p class="text-center text-muted">Sensor + ML (real-time)</p>

<div id="status" class="alert alert-secondary text-center fw-bold">
Status: WAITING FOR SENSOR
</div>
<div id="prob" class="text-center text-muted fw-semibold mb-3"></div>

<div class="row">
<div class="col-md-6">
<div class="card shadow">
<div class="card-body">
<h4 id="bpm">0 BPM</h4>
<canvas id="bpmChart"></canvas>
</div>
</div>
</div>

<div class="col-md-6">
<div class="card shadow">
<div class="card-body">
<h4 id="spo2">0 %</h4>
<canvas id="spo2Chart"></canvas>
</div>
</div>
</div>
</div>
</div>

<!-- 🔊 ALARM AUDIO -->
<audio id="alarmSound">
  <source src="/static/alarm.mp3" type="audio/mpeg">
</audio>

<script>
const evt = new EventSource("/stream");
let alarmPlayed = false;
const alarm = document.getElementById("alarmSound");

const bpmChart = new Chart(document.getElementById("bpmChart"), {
type:'line',
data:{labels:[],datasets:[{label:'BPM',data:[],borderColor:'red'}]},
options:{animation:false}
});

const spo2Chart = new Chart(document.getElementById("spo2Chart"), {
type:'line',
data:{labels:[],datasets:[{label:'SpO2',data:[],borderColor:'green'}]},
options:{animation:false}
});

evt.onmessage = e => {
const d = JSON.parse(e.data);

document.getElementById("bpm").innerText = d.bpm + " BPM";
document.getElementById("spo2").innerText = d.spo2 + " %";

bpmChart.data.labels = d.time_hist;
bpmChart.data.datasets[0].data = d.bpm_hist;
bpmChart.update();

spo2Chart.data.labels = d.time_hist;
spo2Chart.data.datasets[0].data = d.spo2_hist;
spo2Chart.update();

const st = document.getElementById("status");
const pr = document.getElementById("prob");

if(d.status === "APNEA ALERT"){
  st.className = "alert alert-danger text-center fw-bold";
  st.innerText = "⚠️ APNEA ALERT – TAKE ACTION";
  pr.innerText = "Apnea Probability: " + d.probability + " %";

  if(!alarmPlayed){
    alarm.play();
    alarmPlayed = true;
  }
}
else if(d.status === "APNEA"){
  st.className = "alert alert-danger text-center fw-bold";
  st.innerText = "Status: APNEA";
  pr.innerText = "Apnea Probability: " + d.probability + " %";
  alarmPlayed = false;
}
else if(d.status === "NORMAL"){
  st.className = "alert alert-success text-center fw-bold";
  st.innerText = "Status: NORMAL";
  pr.innerText = "Apnea Probability: " + d.probability + " %";
  alarmPlayed = false;
}
else{
  st.className = "alert alert-secondary text-center fw-bold";
  st.innerText = "Status: WAITING FOR SENSOR";
  pr.innerText = "";
  alarmPlayed = false;
}
}
</script>

</body>
</html>
"""

# ---------------- PREDICT ----------------
predict_html = base_head + """
<div class="container mt-4">
<h2 class="text-center">ML Prediction</h2>
<p class="text-center text-muted">Manual prediction without sensor</p>

<div class="card shadow-sm mx-auto" style="max-width:500px">
<div class="card-body">
<form method="post">
<label>BPM</label>
<input name="bpm" class="form-control mb-3" required>
<label>SpO₂</label>
<input name="spo2" class="form-control mb-3" required>
<button class="btn btn-primary w-100">Predict</button>
</form>

{% if result %}
<div class="alert mt-4 {{ 'alert-danger' if result == 'APNEA' else 'alert-success' }}">
<strong>{{ result }}</strong><br>
Probability: {{ prob }} %
</div>
{% endif %}
</div>
</div>
</div>

</body>
</html>
"""

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        row = get_db().execute(
            "SELECT * FROM users WHERE username=?",
            (request.form["username"],)
        ).fetchone()
        if row and check_password_hash(row["password_hash"], request.form["password"]):
            session["user_id"] = row["id"]
            return redirect("/monitor")
        flash("Invalid username or password")
    return render_template_string(login_html)

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        try:
            get_db().execute(
                "INSERT INTO users(username,password_hash) VALUES (?,?)",
                (request.form["username"], generate_password_hash(request.form["password"]))
            )
            get_db().commit()
            return redirect("/")
        except:
            flash("User already exists")
    return render_template_string(register_html)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/monitor")
@login_required
def monitor():
    return render_template_string(monitor_html)

@app.route("/predict", methods=["GET","POST"])
@login_required
def predict():
    result = prob = None
    if request.method == "POST":
        bpm = float(request.form["bpm"])
        spo2 = float(request.form["spo2"])
        result, prob = ml_predict(bpm, spo2)
    return render_template_string(predict_html, result=result, prob=prob)

@app.route("/stream")
@login_required
def stream():
    return Response(event_stream(), mimetype="text/event-stream")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    if not os.path.exists(DATABASE):
        open(DATABASE, "a").close()
    with app.app_context():
        init_db()

    threading.Thread(target=sensor_worker, daemon=True).start()
    app.run(debug=False, use_reloader=False)
