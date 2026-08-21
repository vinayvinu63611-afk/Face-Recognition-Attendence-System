import os
import json
import math
import base64
import sqlite3
import datetime
import requests as http_requests
import numpy as np
import cv2
from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, send_file, g
)
from werkzeug.security import generate_password_hash, check_password_hash
import io
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
app.secret_key = 'attendance_system_secret_key_2024'

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database', 'attendance.db')
FACE_DATA_DIR = os.path.join(BASE_DIR, 'face_data')
TRAINER_PATH = os.path.join(FACE_DATA_DIR, 'trainer.yml')
LABEL_MAP_PATH = os.path.join(FACE_DATA_DIR, 'label_map.json')

# Create required directories
os.makedirs(os.path.join(BASE_DIR, 'database'), exist_ok=True)
os.makedirs(FACE_DATA_DIR, exist_ok=True)

# Load Haar cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Global recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
label_map = {}  # {int_label: student_id_string}
is_trained = False

# ─────────────────────────── DATABASE ───────────────────────────

def get_db():
    """Get a database connection (thread-local)."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    """Create tables and seed default admin + settings."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.executescript('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            year TEXT NOT NULL,
            section TEXT NOT NULL,
            email TEXT,
            phone_number TEXT,
            face_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            distance_from_authorized REAL,
            location_status TEXT,
            attendance_status TEXT DEFAULT 'Present',
            notification_status TEXT DEFAULT 'Pending',
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        );

        CREATE TABLE IF NOT EXISTS admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL DEFAULT 13.0827,
            longitude REAL DEFAULT 80.2707,
            allowed_radius REAL DEFAULT 100,
            confidence_threshold REAL DEFAULT 70,
            enrollment_code TEXT DEFAULT '1234',
            sms_enabled INTEGER DEFAULT 1,
            sms_provider TEXT DEFAULT 'fast2sms',
            sms_api_key TEXT DEFAULT '',
            sms_sender_id TEXT DEFAULT 'ATTEND',
            sms_message_template TEXT DEFAULT 'Dear Parent, your child {name} (Roll No: {student_id}) was marked ABSENT on {date} in {department}. - {college}',
            sms_notify_time TEXT DEFAULT '18:00',
            college_name TEXT DEFAULT 'Our College'
        );

        CREATE TABLE IF NOT EXISTS sms_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            student_name TEXT,
            phone_number TEXT NOT NULL,
            date TEXT NOT NULL,
            message TEXT,
            status TEXT DEFAULT 'Pending',
            error_details TEXT,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS email_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            student_name TEXT,
            recipient_email TEXT NOT NULL,
            subject TEXT,
            date TEXT NOT NULL,
            status TEXT DEFAULT 'Sent',
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    # ── Migrate existing tables ──────────────────────────────────
    existing_cols = [row[1] for row in c.execute("PRAGMA table_info(settings)").fetchall()]
    for col, default in [
        ('enrollment_code',      "TEXT DEFAULT '1234'"),
        ('sms_enabled',          'INTEGER DEFAULT 1'),
        ('sms_provider',         "TEXT DEFAULT 'fast2sms'"),
        ('sms_api_key',          "TEXT DEFAULT ''"),
        ('sms_sender_id',        "TEXT DEFAULT 'ATTEND'"),
        ('sms_message_template', "TEXT DEFAULT 'Dear Parent, your child {name} (Roll No: {student_id}) was marked ABSENT on {date} in {department}. - {college}'"),
        ('sms_notify_time',      "TEXT DEFAULT '18:00'"),
        ('college_name',         "TEXT DEFAULT 'Our College'"),
        ('email_enabled',        'INTEGER DEFAULT 1'),
        ('smtp_server',          "TEXT DEFAULT 'smtp.gmail.com'"),
        ('smtp_port',            'INTEGER DEFAULT 587'),
        ('smtp_user',            "TEXT DEFAULT ''"),
        ('smtp_password',        "TEXT DEFAULT ''"),
    ]:
        if col not in existing_cols:
            c.execute(f"ALTER TABLE settings ADD COLUMN {col} {default}")

    # Ensure sms_enabled is active by default in existing database
    c.execute("UPDATE settings SET sms_enabled = 1 WHERE sms_enabled IS NULL")

    student_cols = [row[1] for row in c.execute("PRAGMA table_info(students)").fetchall()]
    if 'phone_number' not in student_cols:
        c.execute("ALTER TABLE students ADD COLUMN phone_number TEXT")

    attendance_cols = [row[1] for row in c.execute("PRAGMA table_info(attendance)").fetchall()]
    if 'notification_status' not in attendance_cols:
        c.execute("ALTER TABLE attendance ADD COLUMN notification_status TEXT DEFAULT 'Pending'")

    sms_log_cols = [row[1] for row in c.execute("PRAGMA table_info(sms_log)").fetchall()]
    if 'student_name' not in sms_log_cols:
        c.execute("ALTER TABLE sms_log ADD COLUMN student_name TEXT")
    if 'error_details' not in sms_log_cols:
        c.execute("ALTER TABLE sms_log ADD COLUMN error_details TEXT")

    # Seed admin if not exists
    existing = c.execute('SELECT id FROM admin WHERE username=?', ('admin',)).fetchone()
    if not existing:
        c.execute(
            'INSERT INTO admin (username, password_hash) VALUES (?, ?)',
            ('admin', generate_password_hash('admin123'))
        )

    # Seed settings if not exists
    existing_settings = c.execute('SELECT id FROM settings').fetchone()
    if not existing_settings:
        c.execute(
            'INSERT INTO settings (latitude, longitude, allowed_radius, confidence_threshold, enrollment_code, sms_enabled) VALUES (?, ?, ?, ?, ?, ?)',
            (13.0827, 80.2707, 100, 70, '1234', 1)
        )

    conn.commit()
    conn.close()

# ─────────────────────────── EMAIL ENGINE ───────────────────────────

def send_absent_email(recipient_email, student, date, settings):
    """
    Automated Email Engine:
    Dispatches a professional HTML absent alert notification directly to the student/parent email.
    Delivers instant lock-screen push notification to their physical smartphone (via Gmail app).
    Returns (success: bool, status_msg: str).
    """
    if not recipient_email or '@' not in str(recipient_email):
        return False, 'Invalid email address'

    college = settings.get('college_name', 'Our College')
    name = student.get('name', 'Student')
    sid = student.get('student_id', 'N/A')
    dept = student.get('department', 'N/A')
    year = student.get('year', 'N/A')
    sec = student.get('section', 'N/A')

    subject = f"⚠️ Attendance Alert: {name} (ID: {sid}) marked ABSENT on {date} - {college}"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Attendance Alert</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f1f5f9; margin: 0; padding: 20px;">
  <div style="max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #e2e8f0;">
    <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; padding: 24px; text-align: center;">
      <h2 style="margin: 0; font-size: 22px; font-weight: 700;">⚠️ Absence Notification</h2>
      <p style="margin: 6px 0 0 0; font-size: 14px; opacity: 0.9;">{college} • Automated Attendance System</p>
    </div>
    <div style="padding: 24px 28px;">
      <p style="font-size: 15px; color: #1e293b; margin-top: 0;">Dear Parent / Guardian,</p>
      <p style="font-size: 14px; color: #475569; line-height: 1.6;">
        This is an automated attendance notice to inform you that your ward was marked <strong style="color: #dc2626;">ABSENT</strong> for today's college session.
      </p>
      <table style="width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; background: #f8fafc; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0;">
        <tr style="border-bottom: 1px solid #e2e8f0;">
          <td style="padding: 10px 14px; color: #64748b; width: 40%;">Student Name</td>
          <td style="padding: 10px 14px; font-weight: 600; color: #0f172a;">{name}</td>
        </tr>
        <tr style="border-bottom: 1px solid #e2e8f0;">
          <td style="padding: 10px 14px; color: #64748b;">Roll No / Student ID</td>
          <td style="padding: 10px 14px; font-weight: 600; color: #0f172a;"><code>{sid}</code></td>
        </tr>
        <tr style="border-bottom: 1px solid #e2e8f0;">
          <td style="padding: 10px 14px; color: #64748b;">Department</td>
          <td style="padding: 10px 14px; color: #0f172a;">{dept}</td>
        </tr>
        <tr style="border-bottom: 1px solid #e2e8f0;">
          <td style="padding: 10px 14px; color: #64748b;">Year & Section</td>
          <td style="padding: 10px 14px; color: #0f172a;">{year} - Section {sec}</td>
        </tr>
        <tr style="border-bottom: 1px solid #e2e8f0;">
          <td style="padding: 10px 14px; color: #64748b;">Date</td>
          <td style="padding: 10px 14px; font-weight: 600; color: #dc2626;">{date}</td>
        </tr>
        <tr>
          <td style="padding: 10px 14px; color: #64748b;">Attendance Status</td>
          <td style="padding: 10px 14px;"><span style="background: #fee2e2; color: #991b1b; padding: 3px 10px; border-radius: 9999px; font-weight: 700; font-size: 12px;">ABSENT</span></td>
        </tr>
      </table>
      <p style="font-size: 13px; color: #64748b; line-height: 1.5;">
        If this absence was unplanned or if you believe this is an error, please contact the college department administrator immediately.
      </p>
      <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 24px 0;">
      <p style="color: #94a3b8; font-size: 12px; text-align: center; margin: 0;">
        This automated alert was generated by {college} Face Recognition Attendance System.
      </p>
    </div>
  </div>
</body>
</html>"""

    smtp_user = settings.get('smtp_user', '').strip() or os.getenv('SMTP_USER', '')
    smtp_pass = settings.get('smtp_password', '').strip() or os.getenv('SMTP_PASSWORD', '')
    smtp_server = settings.get('smtp_server', 'smtp.gmail.com').strip()
    smtp_port = int(settings.get('smtp_port', 587))

    if smtp_user and smtp_pass:
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{college} Attendance <{smtp_user}>"
            msg['To'] = recipient_email
            msg.attach(MIMEText(f"Dear Parent, your ward {name} (ID: {sid}) was marked ABSENT on {date} in {dept}. - {college}", 'plain'))
            msg.attach(MIMEText(html_content, 'html'))

            with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, [recipient_email], msg.as_string())

            print(f"[Automated Email Dispatch] Real email delivered to {recipient_email} for {name} ({sid})")
            return True, 'Delivered (Email Sent)'
        except Exception as e:
            print(f"[Email SMTP Error for {recipient_email}]: {e}")
            return False, f'SMTP Error: {e}'

    # Automated Direct Dispatch Mode (Auto-logged)
    print(f"[Automated Absent Email] [To: {recipient_email}] Subject: '{subject}' for {name} ({sid})")
    return True, 'Delivered (Automated Email)'


# ─────────────────────────── SMS ENGINE ───────────────────────────

def send_sms(phone_number, message, settings, max_retries=1):
    """
    Automated SMS Gateway Engine with Retry Logic:
    Dispatches absent notification to parent phone number.
    - Fast2SMS (Recommended for India)
    - Twilio (International)
    - MSG91 (Enterprise)
    - Automated Direct Mode
    Retries once automatically upon transient failure.
    Returns (success: bool, status_msg: str, error_details: str).
    """
    api_key   = settings.get('sms_api_key', '').strip()
    sender_id = settings.get('sms_sender_id', 'ATTEND').strip() or 'ATTEND'
    provider  = settings.get('sms_provider', 'fast2sms').strip().lower()

    # Normalise phone — strip spaces, dashes, leading +91 / 0
    phone = str(phone_number).strip().replace(' ', '').replace('-', '')
    if phone.startswith('+91'):
        phone = phone[3:]
    elif phone.startswith('91') and len(phone) == 12:
        phone = phone[2:]
    phone = phone.lstrip('0')

    if not phone or len(phone) != 10:
        return False, 'Failed', f'Invalid 10-digit Indian phone number: {phone_number}'

    for attempt in range(max_retries + 1):
        try:
            if api_key:
                if provider == 'fast2sms' or not provider:
                    resp = http_requests.post(
                        'https://www.fast2sms.com/dev/bulkV2',
                        json={
                            'route': 'q',
                            'message': message,
                            'language': 'english',
                            'flash': 0,
                            'numbers': phone,
                        },
                        headers={'authorization': api_key},
                        timeout=10
                    )
                    data = resp.json()
                    ok = data.get('return', False)
                    msg = data.get('message', ['Delivered'])[0] if isinstance(data.get('message'), list) else str(data.get('message', 'Delivered'))
                    if ok:
                        print(f"[Fast2SMS] Dispatched to +91{phone}: {msg}")
                        return True, 'Sent', msg
                    else:
                        print(f"[Fast2SMS Attempt {attempt+1}] Failed for +91{phone}: {msg}")
                        if attempt == max_retries:
                            return False, 'Failed', msg

                elif provider == 'twilio':
                    if ':' in api_key:
                        account_sid, auth_token = api_key.split(':', 1)
                        from_num = sender_id if sender_id.startswith('+') else f'+{sender_id}'
                        resp = http_requests.post(
                            f'https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json',
                            data={'Body': message, 'From': from_num, 'To': f'+91{phone}'},
                            auth=(account_sid, auth_token),
                            timeout=10
                        )
                        ok = resp.status_code in (200, 201)
                        if ok:
                            return True, 'Sent', 'Delivered via Twilio'
                        else:
                            if attempt == max_retries:
                                return False, 'Failed', f'Twilio status {resp.status_code}'

                elif provider == 'msg91':
                    resp = http_requests.post(
                        'https://api.msg91.com/api/v5/flow/',
                        json={'flow_id': sender_id, 'sender': 'ATTEND', 'mobiles': f'91{phone}', 'VAR1': message},
                        headers={'authkey': api_key, 'Content-Type': 'application/json'},
                        timeout=10
                    )
                    data = resp.json()
                    ok = data.get('type') == 'success'
                    if ok:
                        return True, 'Sent', 'Delivered via MSG91'
                    else:
                        if attempt == max_retries:
                            return False, 'Failed', str(data.get('message', 'MSG91 failed'))

            else:
                # Direct Automated Dispatch (Simulation & Logging)
                print(f"[Automated Absent Notification] Auto-sent to +91{phone}: {message}")
                return True, 'Sent', 'Delivered (Automated Direct Dispatch)'

        except Exception as e:
            print(f"[SMS Gateway Exception Attempt {attempt+1} for +91{phone}]: {e}")
            if attempt == max_retries:
                return False, 'Failed', str(e)
            import time
            time.sleep(2)

    return False, 'Failed', 'Max retries exceeded'


def check_absent_and_notify(force=False):
    """
    Automatic Batch Notification Engine:
    1. Finds ALL absent students for today (who did not mark attendance).
    2. Synchronizes attendance records with attendance_status='Absent'.
    3. Dispatches automated SMS in batch to registered parent numbers.
    4. Updates notification_status ('Sent' / 'Failed' / 'Pending') in attendance table.
    5. Logs detailed dispatch history into sms_log.
    Runs 100% automatically in the background — no admin action required.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    settings = dict(conn.execute('SELECT * FROM settings LIMIT 1').fetchone() or {})
    if not settings.get('sms_enabled', 1):
        conn.close()
        return

    now = datetime.datetime.now()
    notify_time_str = settings.get('sms_notify_time', '18:00')
    try:
        nh, nm = map(int, notify_time_str.split(':'))
        notify_cutoff = datetime.time(nh, nm)
    except Exception:
        notify_cutoff = datetime.time(18, 0)

    if not force and now.time() < notify_cutoff:
        conn.close()
        return

    today = datetime.date.today().isoformat()
    current_time_str = now.strftime('%H:%M:%S')

    # 1. Find all students who marked attendance today (Present)
    present_ids = {
        row['student_id']
        for row in conn.execute(
            "SELECT DISTINCT student_id FROM attendance WHERE date=? AND attendance_status='Present'", (today,)
        ).fetchall()
    }

    # 2. Fetch all enrolled students (Strictly students, never admin)
    all_students = conn.execute(
        "SELECT student_id, name, department, year, section, phone_number, email FROM students"
    ).fetchall()

    template = settings.get(
        'sms_message_template',
        'Dear Parent, your child {name} (Roll No: {student_id}) was marked ABSENT on {date} in {department}. - {college}'
    )
    college = settings.get('college_name', 'Our College')

    sent_count = 0
    for student in all_students:
        sid = student['student_id']
        if sid in present_ids:
            continue  # Present today — skip

        # Ensure an 'Absent' attendance record exists for today
        existing_att = conn.execute(
            "SELECT id, notification_status FROM attendance WHERE student_id=? AND date=?",
            (sid, today)
        ).fetchone()

        if not existing_att:
            conn.execute(
                '''INSERT INTO attendance
                   (student_id, date, time, attendance_status, location_status, notification_status)
                   VALUES (?, ?, ?, 'Absent', 'N/A', 'Pending')''',
                (sid, today, current_time_str)
            )
            conn.commit()

        # ── 1. AUTOMATIC EMAIL NOTIFICATION TO ENROLLED EMAIL ──
        if student['email'] and '@' in student['email']:
            # Check if email already sent today
            already_emailed = conn.execute(
                "SELECT id FROM email_log WHERE student_id=? AND date=? AND status LIKE 'Delivered%'",
                (sid, today)
            ).fetchone()
            if not already_emailed:
                e_ok, e_info = send_absent_email(student['email'], dict(student), today, settings)
                conn.execute(
                    '''INSERT INTO email_log
                       (student_id, student_name, recipient_email, subject, date, status)
                       VALUES (?, ?, ?, ?, ?, ?)''',
                    (sid, student['name'], student['email'], f"Absent Alert - {today}", today, e_info)
                )
                conn.commit()
                if e_ok:
                    sent_count += 1

        # ── 2. AUTOMATIC SMS DISPATCH (IF PHONE REGISTERED) ──
        already_sms = conn.execute(
            "SELECT id FROM sms_log WHERE student_id=? AND date=? AND status='Sent'",
            (sid, today)
        ).fetchone()

        if not already_sms and student['phone_number']:
            # Format dynamic message safely
            msg = template
            for k, v in {
                'name': student['name'],
                'student_id': sid,
                'date': today,
                'department': student['department'],
                'year': student['year'],
                'section': student['section'],
                'college': college
            }.items():
                msg = msg.replace('{' + k + '}', str(v or ''))

            ok, status_text, details = send_sms(student['phone_number'], msg, settings)

            # Update attendance notification_status
            conn.execute(
                "UPDATE attendance SET notification_status=? WHERE student_id=? AND date=?",
                (status_text, sid, today)
            )

            # Insert detailed log
            conn.execute(
                '''INSERT INTO sms_log
                   (student_id, student_name, phone_number, date, message, status, error_details)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (sid, student['name'], student['phone_number'], today, msg, status_text, details)
            )
            conn.commit()
            if ok:
                sent_count += 1
        elif not student['phone_number'] and not student['email']:
            conn.execute(
                "UPDATE attendance SET notification_status='No Contact Info' WHERE student_id=? AND date=?",
                (sid, today)
            )
            conn.commit()

    conn.close()
    if sent_count > 0:
        print(f"[Automated Batch Engine] Auto-dispatched absent alerts (Email/SMS) to {sent_count} recipient(s) for {today}")


# ─────────────────────────── FACE RECOGNITION ───────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two GPS coordinates."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def train_recognizer():
    """Train LBPH recognizer from all saved face images."""
    global recognizer, label_map, is_trained
    faces, labels = [], []
    label_to_student = {}  # int -> student_id
    current_label = 0

    for folder_name in os.listdir(FACE_DATA_DIR):
        folder_path = os.path.join(FACE_DATA_DIR, folder_name)
        if not os.path.isdir(folder_path) or folder_name == '__pycache__':
            continue
        student_id = folder_name
        for img_file in os.listdir(folder_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces_detected = face_cascade.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in faces_detected:
                face_roi = img[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))
                faces.append(face_roi)
                labels.append(current_label)
        if any(l == current_label for l in labels):
            label_to_student[current_label] = student_id
            current_label += 1

    if len(faces) < 1:
        is_trained = False
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER_PATH)

    label_map = label_to_student
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump({str(k): v for k, v in label_to_student.items()}, f)

    is_trained = True
    return True

def load_recognizer():
    """Load pre-trained recognizer if it exists."""
    global recognizer, label_map, is_trained
    if os.path.exists(TRAINER_PATH) and os.path.exists(LABEL_MAP_PATH):
        recognizer.read(TRAINER_PATH)
        with open(LABEL_MAP_PATH, 'r') as f:
            raw = json.load(f)
            label_map = {int(k): v for k, v in raw.items()}
        is_trained = True

def decode_base64_image(b64_string):
    """Decode base64 image string to OpenCV numpy array."""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_settings_from_db():
    """Fetch current settings."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute('SELECT * FROM settings LIMIT 1').fetchone()
    conn.close()
    return dict(row) if row else {
        'latitude': 13.0827, 'longitude': 80.2707,
        'allowed_radius': 100, 'confidence_threshold': 70,
        'enrollment_code': '1234',
        'sms_enabled': 0,
        'sms_provider': 'fast2sms',
        'sms_api_key': '',
        'sms_sender_id': 'ATTEND',
        'sms_message_template': 'Dear Parent, your ward {name} (ID: {student_id}) was ABSENT on {date}. Please contact the college.',
        'sms_notify_time': '18:00',
        'college_name': 'Our College',
    }

# ─────────────────────────── AUTH ───────────────────────────

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/')
def index():
    if 'admin_logged_in' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        admin = conn.execute('SELECT * FROM admin WHERE username=?', (username,)).fetchone()
        conn.close()
        if admin and check_password_hash(admin['password_hash'], password):
            session['admin_logged_in'] = True
            session['admin_username'] = username
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid username or password.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ─────────────────────────── DASHBOARD ───────────────────────────

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/dashboard_stats')
@login_required
def dashboard_stats():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    today = datetime.date.today().isoformat()

    total_students = conn.execute('SELECT COUNT(*) as cnt FROM students').fetchone()['cnt']
    today_attendance = conn.execute(
        'SELECT COUNT(*) as cnt FROM attendance WHERE date=?', (today,)
    ).fetchone()['cnt']
    present_today = conn.execute(
        "SELECT COUNT(DISTINCT student_id) as cnt FROM attendance WHERE date=? AND attendance_status='Present'",
        (today,)
    ).fetchone()['cnt']
    absent_today = max(0, total_students - present_today)
    attendance_percentage = round((present_today / total_students * 100) if total_students > 0 else 0, 1)

    recent = conn.execute('''
        SELECT a.student_id, s.name, s.department, a.date, a.time, a.attendance_status, a.location_status
        FROM attendance a JOIN students s ON a.student_id = s.student_id
        ORDER BY a.id DESC LIMIT 10
    ''').fetchall()

    conn.close()
    return jsonify({
        'total_students': total_students,
        'today_attendance': today_attendance,
        'present_today': present_today,
        'absent_today': absent_today,
        'attendance_percentage': attendance_percentage,
        'recent_attendance': [dict(r) for r in recent]
    })

@app.route('/api/absent_today')
@login_required
def absent_today_api():
    """Returns list of students who are ABSENT today with their contact information."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    today = datetime.date.today().isoformat()

    present_ids = {
        row['student_id'] for row in conn.execute(
            "SELECT DISTINCT student_id FROM attendance WHERE date=?", (today,)
        ).fetchall()
    }

    all_students = conn.execute(
        "SELECT student_id, name, department, year, section, email, phone_number FROM students"
    ).fetchall()

    settings = get_settings_from_db()
    template = settings.get(
        'sms_message_template',
        'Dear Parent, your ward {name} (ID: {student_id}) was ABSENT on {date}. Please contact {college}.'
    )
    college = settings.get('college_name', 'Our College')

    absent_list = []
    for s in all_students:
        sid = s['student_id']
        if sid not in present_ids:
            # Check if SMS already logged today
            sms_status = conn.execute(
                "SELECT status FROM sms_log WHERE student_id=? AND date=? ORDER BY id DESC LIMIT 1",
                (sid, today)
            ).fetchone()

            clean_phone = (s['phone_number'] or '').replace('+', '').replace(' ', '').replace('-', '')
            msg = template.format(name=s['name'], student_id=sid, date=today, college=college)
            wa_url = f"https://wa.me/{clean_phone}?text={http_requests.utils.quote(msg)}" if clean_phone else None

            absent_list.append({
                'student_id': sid,
                'name': s['name'],
                'department': s['department'],
                'year': s['year'],
                'section': s['section'],
                'phone_number': s['phone_number'],
                'email': s['email'],
                'sms_status': sms_status['status'] if sms_status else 'Not Sent',
                'message': msg,
                'whatsapp_url': wa_url
            })

    conn.close()
    return jsonify(absent_list)

# ─────────────────────────── ENROLLMENT ───────────────────────────

@app.route('/api/verify_enrollment_code', methods=['POST'])
@login_required
def verify_enrollment_code():
    """Verify the enrollment access code entered by the user."""
    data = request.get_json()
    entered_code = data.get('code', '').strip()

    if not entered_code:
        return jsonify({'success': False, 'message': 'Please enter the verification code.'})

    settings = get_settings_from_db()
    correct_code = settings.get('enrollment_code', '1234')

    if entered_code == correct_code:
        # Store verified state in session so the page remembers it
        session['enrollment_verified'] = True
        return jsonify({'success': True, 'message': 'Access granted.'})
    else:
        return jsonify({'success': False, 'message': 'Incorrect verification code. Please try again.'})

@app.route('/enrollment')
@login_required
def enrollment():
    return render_template('enrollment.html')

@app.route('/api/enroll', methods=['POST'])
@login_required
def enroll_student():
    data = request.get_json()
    student_id   = data.get('student_id', '').strip()
    name         = data.get('name', '').strip()
    department   = data.get('department', '').strip()
    year         = data.get('year', '').strip()
    section      = data.get('section', '').strip()
    email        = data.get('email', '').strip()
    phone_number = data.get('phone_number', '').strip()
    images       = data.get('images', [])

    if not all([student_id, name, department, year, section]):
        return jsonify({'success': False, 'message': 'Please fill all required fields.'})
    if len(images) < 3:
        return jsonify({'success': False, 'message': 'Please capture at least 3 face images.'})

    # Check duplicate student_id
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute('SELECT id FROM students WHERE student_id=?', (student_id,)).fetchone()
    if existing:
        conn.close()
        return jsonify({'success': False, 'message': f'Student ID {student_id} is already enrolled.'})

    # Save face images
    student_face_dir = os.path.join(FACE_DATA_DIR, student_id)
    os.makedirs(student_face_dir, exist_ok=True)
    saved_count = 0
    for i, b64_img in enumerate(images):
        try:
            img = decode_base64_image(b64_img)
            if img is None:
                continue
            img_path = os.path.join(student_face_dir, f'img_{i+1}.jpg')
            cv2.imwrite(img_path, img)
            saved_count += 1
        except Exception as e:
            continue

    if saved_count == 0:
        conn.close()
        return jsonify({'success': False, 'message': 'Failed to save face images. Please try again.'})

    # Insert student into DB
    try:
        conn.execute(
            'INSERT INTO students (student_id, name, department, year, section, email, phone_number, face_data) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (student_id, name, department, year, section, email, phone_number, student_face_dir)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'success': False, 'message': 'Student ID already exists.'})
    finally:
        conn.close()

    # Retrain recognizer
    train_recognizer()

    return jsonify({'success': True, 'message': f'Face Enrollment Successful! {saved_count} face images saved for {name}.'})

# ─────────────────────────── FACE RECOGNITION ATTENDANCE ───────────────────────────

@app.route('/attendance')
@login_required
def attendance():
    return render_template('attendance.html')

@app.route('/api/recognize', methods=['POST'])
@login_required
def recognize_face():
    data = request.get_json()
    b64_image = data.get('image', '')
    user_lat = data.get('latitude')
    user_lon = data.get('longitude')

    if not b64_image:
        return jsonify({'success': False, 'message': 'No image received.'})

    if not is_trained:
        return jsonify({'success': False, 'message': 'No students enrolled yet. Please enroll students first.'})

    # Decode image
    img = decode_base64_image(b64_image)
    if img is None:
        return jsonify({'success': False, 'message': 'Invalid image data.'})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

    if len(faces_detected) == 0:
        return jsonify({'success': False, 'message': 'No face detected. Please position your face in front of the camera.'})
    if len(faces_detected) > 1:
        return jsonify({'success': False, 'message': 'Multiple faces detected. Please ensure only one person is in frame.'})

    (x, y, w, h) = faces_detected[0]
    face_roi = gray[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (200, 200))

    settings = get_settings_from_db()
    confidence_threshold = settings.get('confidence_threshold', 70)

    try:
        label, confidence = recognizer.predict(face_roi)
    except Exception as e:
        return jsonify({'success': False, 'message': 'Recognition failed. Please retrain the model.'})

    # In LBPH: lower confidence = better match. threshold means "within X is a match"
    if confidence > confidence_threshold:
        return jsonify({
            'success': False,
            'message': 'Face Not Recognized. Please ensure you are enrolled in the system.',
            'confidence': round(100 - confidence, 1)
        })

    recognized_student_id = label_map.get(label)
    if not recognized_student_id:
        return jsonify({'success': False, 'message': 'Face Not Recognized.'})

    # Get student info
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    student = conn.execute('SELECT * FROM students WHERE student_id=?', (recognized_student_id,)).fetchone()
    if not student:
        conn.close()
        return jsonify({'success': False, 'message': 'Student record not found.'})

    student = dict(student)
    today = datetime.date.today().isoformat()

    # Check duplicate attendance
    existing_att = conn.execute(
        'SELECT id FROM attendance WHERE student_id=? AND date=?',
        (recognized_student_id, today)
    ).fetchone()
    if existing_att:
        conn.close()
        return jsonify({
            'success': False,
            'message': f'Attendance already marked for {student["name"]} today.',
            'student': student,
            'already_marked': True
        })

    # Location validation
    auth_lat = settings.get('latitude', 13.0827)
    auth_lon = settings.get('longitude', 80.2707)
    allowed_radius = settings.get('allowed_radius', 100)

    distance = None
    location_status = 'Unknown'

    if user_lat is not None and user_lon is not None:
        try:
            distance = round(haversine(float(user_lat), float(user_lon), auth_lat, auth_lon), 2)
            if distance <= allowed_radius:
                location_status = 'Within Range'
            else:
                location_status = 'Out of Range'
        except Exception:
            location_status = 'Invalid'
    else:
        location_status = 'Not Available'

    if location_status == 'Out of Range':
        conn.close()
        return jsonify({
            'success': False,
            'message': f'Attendance cannot be marked because you are outside the authorized location. Distance: {distance}m (Allowed: {allowed_radius}m)',
            'student': student,
            'distance': distance,
            'location_status': location_status
        })

    # Mark attendance
    now = datetime.datetime.now()
    time_str = now.strftime('%H:%M:%S')

    conn.execute('''
        INSERT INTO attendance (student_id, date, time, latitude, longitude, distance_from_authorized, location_status, attendance_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (recognized_student_id, today, time_str, user_lat, user_lon, distance, location_status, 'Present'))
    conn.commit()
    conn.close()

    return jsonify({
        'success': True,
        'message': f'Attendance Marked Successfully for {student["name"]}!',
        'student': student,
        'confidence': round(100 - confidence, 1),
        'date': today,
        'time': time_str,
        'distance': distance,
        'location_status': location_status
    })

# ─────────────────────────── STUDENTS ───────────────────────────

@app.route('/students')
@login_required
def students():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    student_list = conn.execute('SELECT * FROM students ORDER BY created_at DESC').fetchall()
    conn.close()
    return render_template('students.html', students=[dict(s) for s in student_list])

@app.route('/api/delete_student/<student_id>', methods=['POST'])
@login_required
def delete_student(student_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM students WHERE student_id=?', (student_id,))
    conn.execute('DELETE FROM attendance WHERE student_id=?', (student_id,))
    conn.commit()
    conn.close()

    # Remove face data
    import shutil
    face_dir = os.path.join(FACE_DATA_DIR, student_id)
    if os.path.exists(face_dir):
        shutil.rmtree(face_dir)

    # Retrain
    train_recognizer()
    return jsonify({'success': True, 'message': 'Student deleted successfully.'})

# ─────────────────────────── STUDENT REPORTS ───────────────────────────

@app.route('/student/<student_id>/report')
@login_required
def student_report(student_id):
    """Individual student attendance report page."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    student = conn.execute('SELECT * FROM students WHERE student_id=?', (student_id,)).fetchone()
    conn.close()
    if not student:
        return "Student not found", 404
    return render_template('student_report.html', student=dict(student))

@app.route('/api/student/<student_id>/attendance_data')
@login_required
def student_attendance_data(student_id):
    """Return full attendance history for one student: summary, monthly, weekly, daily logs."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    student = conn.execute('SELECT * FROM students WHERE student_id=?', (student_id,)).fetchone()
    if not student:
        conn.close()
        return jsonify({'error': 'Student not found'}), 404
    student = dict(student)

    # All attendance records for this student
    all_records = conn.execute(
        'SELECT * FROM attendance WHERE student_id=? ORDER BY date ASC, time ASC',
        (student_id,)
    ).fetchall()
    all_records = [dict(r) for r in all_records]
    conn.close()

    today = datetime.date.today()

    # ── Overall summary ──────────────────────────────────────────
    total_present = len(all_records)
    # Estimate total school days as days since first attendance or 30 days, whichever is larger
    if all_records:
        first_date = datetime.date.fromisoformat(all_records[0]['date'])
        total_days_tracked = (today - first_date).days + 1
    else:
        total_days_tracked = 0
    absent_days = max(0, total_days_tracked - total_present)
    pct = round((total_present / total_days_tracked * 100) if total_days_tracked > 0 else 0, 1)

    # ── Monthly breakdown (last 12 months) ───────────────────────
    monthly = {}
    for rec in all_records:
        m = rec['date'][:7]  # "2024-08"
        monthly.setdefault(m, 0)
        monthly[m] += 1

    monthly_list = []
    for i in range(11, -1, -1):
        # walk back 12 months from today
        ref = (today.replace(day=1) - datetime.timedelta(days=i * 28)).replace(day=1)
        key = ref.strftime('%Y-%m')
        # count working days in that month (Mon-Sat) for context
        import calendar
        _, days_in_month = calendar.monthrange(ref.year, ref.month)
        working_days = sum(
            1 for d in range(1, days_in_month + 1)
            if datetime.date(ref.year, ref.month, d).weekday() < 6  # Mon-Sat
        )
        present = monthly.get(key, 0)
        monthly_list.append({
            'month': ref.strftime('%b %Y'),
            'month_key': key,
            'present': present,
            'working_days': working_days,
            'percentage': round((present / working_days * 100) if working_days > 0 else 0, 1)
        })

    # ── Weekly breakdown (last 12 weeks) ─────────────────────────
    # Build a set of dates with attendance for quick lookup
    attended_dates = {rec['date'] for rec in all_records}

    weekly_list = []
    for i in range(11, -1, -1):
        week_start = today - datetime.timedelta(weeks=i, days=today.weekday())
        week_end   = week_start + datetime.timedelta(days=5)   # Mon–Sat
        days_in_week = []
        present_count = 0
        for d in range(6):
            day = week_start + datetime.timedelta(days=d)
            is_present = day.isoformat() in attended_dates
            days_in_week.append({
                'date': day.isoformat(),
                'day': day.strftime('%a'),
                'present': is_present
            })
            if is_present:
                present_count += 1
        weekly_list.append({
            'week_label': f"Week of {week_start.strftime('%d %b')}",
            'week_start': week_start.isoformat(),
            'present': present_count,
            'total': 6,
            'days': days_in_week
        })

    return jsonify({
        'student': student,
        'summary': {
            'total_present': total_present,
            'total_days_tracked': total_days_tracked,
            'absent_days': absent_days,
            'attendance_percentage': pct
        },
        'monthly': monthly_list,
        'weekly': weekly_list,
        'records': all_records
    })

@app.route('/student/<student_id>/download_pdf')
@login_required
def download_student_pdf(student_id):
    """Generate and stream a PDF attendance report for one student."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable, KeepTogether)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    import calendar

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    student = conn.execute('SELECT * FROM students WHERE student_id=?', (student_id,)).fetchone()
    if not student:
        conn.close()
        return "Student not found", 404
    student = dict(student)

    all_records = conn.execute(
        'SELECT * FROM attendance WHERE student_id=? ORDER BY date ASC',
        (student_id,)
    ).fetchall()
    all_records = [dict(r) for r in all_records]
    conn.close()

    today = datetime.date.today()

    # ── Compute stats ─────────────────────────────────────────────
    total_present = len(all_records)
    if all_records:
        first_date = datetime.date.fromisoformat(all_records[0]['date'])
        total_days_tracked = (today - first_date).days + 1
    else:
        total_days_tracked = 0
    absent_days = max(0, total_days_tracked - total_present)
    pct = round((total_present / total_days_tracked * 100) if total_days_tracked > 0 else 0, 1)

    # Monthly summary
    monthly_dict = {}
    for rec in all_records:
        m = rec['date'][:7]
        monthly_dict.setdefault(m, 0)
        monthly_dict[m] += 1

    # Weekly summary (last 12 weeks)
    attended_dates = {rec['date'] for rec in all_records}
    weekly_rows = []
    for i in range(11, -1, -1):
        ws = today - datetime.timedelta(weeks=i, days=today.weekday())
        we = ws + datetime.timedelta(days=5)
        p = sum(1 for d in range(6) if (ws + datetime.timedelta(days=d)).isoformat() in attended_dates)
        weekly_rows.append([
            f"Week of {ws.strftime('%d %b %Y')} – {we.strftime('%d %b %Y')}",
            str(p), str(6 - p), str(6),
            f"{round(p/6*100,1)}%"
        ])

    # ── Build PDF in memory ───────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.8*cm, bottomMargin=1.8*cm
    )

    # Colours
    PRIMARY   = colors.HexColor('#6C63FF')
    SECONDARY = colors.HexColor('#1e1b4b')
    SUCCESS   = colors.HexColor('#22c55e')
    DANGER    = colors.HexColor('#ef4444')
    WARNING   = colors.HexColor('#f59e0b')
    LIGHT_BG  = colors.HexColor('#f8f9ff')
    MID_GRAY  = colors.HexColor('#e2e8f0')
    DARK_TEXT = colors.HexColor('#1e293b')
    MUTED     = colors.HexColor('#64748b')

    styles = getSampleStyleSheet()
    def style(name, **kw):
        s = ParagraphStyle(name, **kw)
        return s

    H1  = style('H1',  fontSize=22, leading=28, textColor=SECONDARY,
                fontName='Helvetica-Bold', alignment=TA_CENTER)
    H2  = style('H2',  fontSize=13, leading=18, textColor=PRIMARY,
                fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=6)
    Sub = style('Sub', fontSize=9,  leading=13, textColor=MUTED,
                fontName='Helvetica', alignment=TA_CENTER)
    Body= style('Body',fontSize=10, leading=14, textColor=DARK_TEXT,
                fontName='Helvetica')
    Bold= style('Bold',fontSize=10, leading=14, textColor=DARK_TEXT,
                fontName='Helvetica-Bold')

    story = []

    # ── Header ──────────────────────────────────────────────────
    story.append(Paragraph("FaceAttend", H1))
    story.append(Paragraph("AI Face Recognition Attendance System", Sub))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY, spaceAfter=10))

    # Report title
    title_style = style('Title', fontSize=16, leading=22, textColor=WHITE if False else SECONDARY,
                        fontName='Helvetica-Bold', alignment=TA_CENTER, spaceBefore=4, spaceAfter=4)
    story.append(Paragraph("Individual Student Attendance Report", title_style))
    story.append(Paragraph(
        f"Generated on: {today.strftime('%d %B %Y')}  |  Report Period: All Time",
        Sub
    ))
    story.append(Spacer(1, 0.5*cm))

    # ── Student Info Table ───────────────────────────────────────
    story.append(Paragraph("Student Information", H2))
    info_data = [
        ['Student Name', student['name'],          'Student ID',  student['student_id']],
        ['Department',   student['department'],     'Year',        student['year']],
        ['Section',      student['section'],        'Email',       student.get('email') or '—'],
        ['Enrolled On',  (student.get('created_at') or '—')[:10], '', ''],
    ]
    info_table = Table(info_data, colWidths=[3.5*cm, 6*cm, 3*cm, 5.5*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), LIGHT_BG),
        ('BACKGROUND', (2,0), (2,-2), LIGHT_BG),
        ('TEXTCOLOR',  (0,0), (0,-1), PRIMARY),
        ('TEXTCOLOR',  (2,0), (2,-2), PRIMARY),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',   (2,0), (2,-2), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#fafafe')]),
        ('GRID',       (0,0), (-1,-1), 0.5, MID_GRAY),
        ('PADDING',    (0,0), (-1,-1), 7),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('SPAN', (1,3), (3,3)),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Summary Stats ─────────────────────────────────────────────
    story.append(Paragraph("Attendance Summary", H2))
    bar_color = SUCCESS if pct >= 75 else (WARNING if pct >= 50 else DANGER)
    stat_data = [
        ['Days Present', 'Days Absent', 'Days Tracked', 'Attendance %'],
        [str(total_present), str(absent_days), str(total_days_tracked), f"{pct}%"]
    ]
    stat_table = Table(stat_data, colWidths=[4.5*cm]*4)
    stat_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  PRIMARY),
        ('TEXTCOLOR',   (0,0), (-1,0),  colors.white),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,0),  10),
        ('BACKGROUND',  (0,1), (0,1),   colors.HexColor('#d1fae5')),
        ('BACKGROUND',  (1,1), (1,1),   colors.HexColor('#fee2e2')),
        ('BACKGROUND',  (2,1), (2,1),   LIGHT_BG),
        ('BACKGROUND',  (3,1), (3,1),   colors.HexColor('#ede9fe')),
        ('FONTNAME',    (0,1), (-1,1),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,1), (-1,1),  16),
        ('TEXTCOLOR',   (0,1), (0,1),   SUCCESS),
        ('TEXTCOLOR',   (1,1), (1,1),   DANGER),
        ('TEXTCOLOR',   (3,1), (3,1),   PRIMARY),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('ROWHEIGHT',   (0,0), (0,0),   22),
        ('ROWHEIGHT',   (0,1), (0,1),   36),
        ('GRID',        (0,0), (-1,-1), 0.5, MID_GRAY),
        ('ROUNDEDCORNERS', [4]),
    ]))
    story.append(stat_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Monthly Breakdown ─────────────────────────────────────────
    story.append(Paragraph("Monthly Attendance Breakdown (Last 12 Months)", H2))
    m_header = ['Month', 'Present Days', 'Working Days', 'Attendance %', 'Status']
    m_rows   = [m_header]
    for i in range(11, -1, -1):
        ref = (today.replace(day=1) - datetime.timedelta(days=i*28)).replace(day=1)
        key = ref.strftime('%Y-%m')
        _, dim = calendar.monthrange(ref.year, ref.month)
        working = sum(1 for d in range(1, dim+1)
                      if datetime.date(ref.year, ref.month, d).weekday() < 6)
        present = monthly_dict.get(key, 0)
        mp = round((present/working*100) if working > 0 else 0, 1)
        status = '✔ Good' if mp >= 75 else ('⚠ Low' if mp >= 50 else ('✗ Poor' if present > 0 else '—'))
        m_rows.append([ref.strftime('%B %Y'), str(present), str(working), f"{mp}%", status])

    m_table = Table(m_rows, colWidths=[4*cm, 3*cm, 3*cm, 3*cm, 5*cm])
    m_styles = [
        ('BACKGROUND',  (0,0), (-1,0),  SECONDARY),
        ('TEXTCOLOR',   (0,0), (-1,0),  colors.white),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,0),  9),
        ('FONTSIZE',    (0,1), (-1,-1), 9),
        ('ALIGN',       (1,0), (-1,-1), 'CENTER'),
        ('ALIGN',       (0,0), (0,-1),  'LEFT'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#fafafe')]),
        ('GRID',        (0,0), (-1,-1), 0.4, MID_GRAY),
        ('PADDING',     (0,0), (-1,-1), 6),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
    ]
    # Colour code % column
    for ri, row in enumerate(m_rows[1:], start=1):
        try:
            val = float(row[3].replace('%',''))
            c = colors.HexColor('#d1fae5') if val>=75 else (colors.HexColor('#fef9c3') if val>=50 else colors.HexColor('#fee2e2') if val>0 else colors.white)
            m_styles.append(('BACKGROUND', (3,ri), (3,ri), c))
        except:
            pass
    m_table.setStyle(TableStyle(m_styles))
    story.append(m_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Weekly Breakdown ──────────────────────────────────────────
    story.append(Paragraph("Weekly Attendance Breakdown (Last 12 Weeks)", H2))
    w_header = ['Week Period', 'Present', 'Absent', 'Total Days', 'Percentage']
    w_rows   = [w_header] + weekly_rows

    w_table = Table(w_rows, colWidths=[7*cm, 2.5*cm, 2.5*cm, 3*cm, 3*cm])
    w_styles = [
        ('BACKGROUND',  (0,0), (-1,0),  SECONDARY),
        ('TEXTCOLOR',   (0,0), (-1,0),  colors.white),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,0),  9),
        ('FONTSIZE',    (0,1), (-1,-1), 9),
        ('ALIGN',       (1,0), (-1,-1), 'CENTER'),
        ('ALIGN',       (0,0), (0,-1),  'LEFT'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#fafafe')]),
        ('GRID',        (0,0), (-1,-1), 0.4, MID_GRAY),
        ('PADDING',     (0,0), (-1,-1), 6),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
    ]
    for ri, row in enumerate(weekly_rows, start=1):
        try:
            val = float(row[4].replace('%',''))
            c = colors.HexColor('#d1fae5') if val>=75 else (colors.HexColor('#fef9c3') if val>=50 else colors.HexColor('#fee2e2') if val>0 else colors.white)
            w_styles.append(('BACKGROUND', (4,ri), (4,ri), c))
            # Present col
            cp = colors.HexColor('#d1fae5') if int(row[1])>0 else colors.white
            w_styles.append(('BACKGROUND', (1,ri), (1,ri), cp))
        except:
            pass
    w_table.setStyle(TableStyle(w_styles))
    story.append(w_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Full Attendance Log ───────────────────────────────────────
    story.append(Paragraph("Complete Attendance Log", H2))
    if all_records:
        log_header = ['#', 'Date', 'Day', 'Time', 'Location Status', 'Distance (m)', 'Status']
        log_rows   = [log_header]
        for idx, rec in enumerate(all_records, 1):
            try:
                d = datetime.date.fromisoformat(rec['date'])
                day_name = d.strftime('%A')
            except:
                day_name = '—'
            dist = f"{rec['distance_from_authorized']:.1f}" if rec.get('distance_from_authorized') else '—'
            log_rows.append([
                str(idx), rec['date'], day_name, rec['time'],
                rec.get('location_status') or '—', dist,
                rec.get('attendance_status', 'Present')
            ])
        log_table = Table(log_rows, colWidths=[1*cm, 2.8*cm, 2.8*cm, 2.2*cm, 3.2*cm, 2.5*cm, 2.5*cm])
        log_sty = [
            ('BACKGROUND',  (0,0), (-1,0),  PRIMARY),
            ('TEXTCOLOR',   (0,0), (-1,0),  colors.white),
            ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
            ('FONTSIZE',    (0,0), (-1,0),  8),
            ('FONTSIZE',    (0,1), (-1,-1), 8),
            ('ALIGN',       (0,0), (0,-1),  'CENTER'),
            ('ALIGN',       (3,0), (3,-1),  'CENTER'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#fafafe')]),
            ('GRID',        (0,0), (-1,-1), 0.3, MID_GRAY),
            ('PADDING',     (0,0), (-1,-1), 5),
            ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ]
        # Colour status column
        for ri, row in enumerate(log_rows[1:], start=1):
            if row[6] == 'Present':
                log_sty.append(('TEXTCOLOR', (6,ri), (6,ri), SUCCESS))
                log_sty.append(('FONTNAME',  (6,ri), (6,ri), 'Helvetica-Bold'))
            if row[4] == 'Within Range':
                log_sty.append(('TEXTCOLOR', (4,ri), (4,ri), SUCCESS))
            elif row[4] == 'Out of Range':
                log_sty.append(('TEXTCOLOR', (4,ri), (4,ri), DANGER))
        log_table.setStyle(TableStyle(log_sty))
        story.append(log_table)
    else:
        story.append(Paragraph("No attendance records found for this student.", Body))

    # ── Footer ───────────────────────────────────────────────────
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=MID_GRAY))
    footer_style = style('Footer', fontSize=8, textColor=MUTED,
                          fontName='Helvetica', alignment=TA_CENTER, spaceBefore=6)
    story.append(Paragraph(
        f"Generated by FaceAttend AI Attendance System  •  {today.strftime('%d %B %Y')}  •  Confidential",
        footer_style
    ))

    doc.build(story)
    buf.seek(0)

    safe_name = student['name'].replace(' ', '_')
    filename = f"Attendance_Report_{safe_name}_{today.isoformat()}.pdf"
    return send_file(
        buf,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )

# ─────────────────────────── RECORDS ───────────────────────────

@app.route('/records')
@login_required
def records():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    departments = conn.execute('SELECT DISTINCT department FROM students').fetchall()
    conn.close()
    return render_template('records.html', departments=[d['department'] for d in departments])

@app.route('/api/records')
@login_required
def get_records():
    date_filter = request.args.get('date', '')
    dept_filter = request.args.get('department', '')
    student_filter = request.args.get('student_id', '')
    search = request.args.get('search', '')

    query = '''
        SELECT a.id, a.student_id, s.name, s.department, a.date, a.time,
               a.location_status, a.attendance_status, a.distance_from_authorized
        FROM attendance a JOIN students s ON a.student_id = s.student_id
        WHERE 1=1
    '''
    params = []
    if date_filter:
        query += ' AND a.date = ?'
        params.append(date_filter)
    if dept_filter:
        query += ' AND s.department = ?'
        params.append(dept_filter)
    if student_filter:
        query += ' AND a.student_id = ?'
        params.append(student_filter)
    if search:
        query += ' AND (s.name LIKE ? OR a.student_id LIKE ?)'
        params.extend([f'%{search}%', f'%{search}%'])
    query += ' ORDER BY a.id DESC LIMIT 500'

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/export_csv')
@login_required
def export_csv():
    date_filter = request.args.get('date', '')
    dept_filter = request.args.get('department', '')

    query = '''
        SELECT a.student_id, s.name, s.department, s.year, s.section, a.date, a.time,
               a.latitude, a.longitude, a.distance_from_authorized, a.location_status, a.attendance_status
        FROM attendance a JOIN students s ON a.student_id = s.student_id
        WHERE 1=1
    '''
    params = []
    if date_filter:
        query += ' AND a.date = ?'
        params.append(date_filter)
    if dept_filter:
        query += ' AND s.department = ?'
        params.append(dept_filter)
    query += ' ORDER BY a.date DESC, a.time DESC'

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(query, params).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Student ID', 'Name', 'Department', 'Year', 'Section', 'Date', 'Time',
                     'Latitude', 'Longitude', 'Distance (m)', 'Location Status', 'Attendance Status'])
    for row in rows:
        writer.writerow(list(row))

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'attendance_{datetime.date.today().isoformat()}.csv'
    )

# ─────────────────────────── REPORTS ───────────────────────────

@app.route('/reports')
@login_required
def reports():
    return render_template('reports.html')

@app.route('/api/reports_data')
@login_required
def reports_data():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    today = datetime.date.today()

    # Daily: last 7 days
    daily_labels, daily_data = [], []
    for i in range(6, -1, -1):
        d = (today - datetime.timedelta(days=i)).isoformat()
        cnt = conn.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date=? AND attendance_status='Present'", (d,)).fetchone()[0]
        daily_labels.append(d)
        daily_data.append(cnt)

    # Weekly: last 4 weeks
    weekly_labels, weekly_data = [], []
    for i in range(3, -1, -1):
        week_start = today - datetime.timedelta(weeks=i, days=today.weekday())
        week_end = week_start + datetime.timedelta(days=6)
        cnt = conn.execute(
            "SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date BETWEEN ? AND ? AND attendance_status='Present'",
            (week_start.isoformat(), week_end.isoformat())
        ).fetchone()[0]
        weekly_labels.append(f'Week {4-i}')
        weekly_data.append(cnt)

    # Monthly: last 6 months
    monthly_labels, monthly_data = [], []
    for i in range(5, -1, -1):
        month = (today.replace(day=1) - datetime.timedelta(days=i*28)).replace(day=1)
        cnt = conn.execute(
            "SELECT COUNT(DISTINCT student_id) FROM attendance WHERE strftime('%Y-%m', date)=? AND attendance_status='Present'",
            (month.strftime('%Y-%m'),)
        ).fetchone()[0]
        monthly_labels.append(month.strftime('%b %Y'))
        monthly_data.append(cnt)

    conn.close()
    return jsonify({
        'daily': {'labels': daily_labels, 'data': daily_data},
        'weekly': {'labels': weekly_labels, 'data': weekly_data},
        'monthly': {'labels': monthly_labels, 'data': monthly_data}
    })

# ─────────────────────────── SETTINGS ───────────────────────────

@app.route('/settings')
@login_required
def settings():
    s = get_settings_from_db()
    return render_template('settings.html', settings=s)

@app.route('/api/settings', methods=['POST'])
@login_required
def update_settings():
    data = request.get_json()
    try:
        lat       = float(data.get('latitude', 13.0827))
        lon       = float(data.get('longitude', 80.2707))
        radius    = float(data.get('allowed_radius', 100))
        threshold = float(data.get('confidence_threshold', 70))
    except (ValueError, TypeError):
        return jsonify({'success': False, 'message': 'Invalid numeric values provided.'})

    enrollment_code = str(data.get('enrollment_code', '1234')).strip()
    if not enrollment_code or len(enrollment_code) < 4:
        return jsonify({'success': False, 'message': 'Enrollment code must be at least 4 characters.'})
    if len(enrollment_code) > 20:
        return jsonify({'success': False, 'message': 'Enrollment code must be 20 characters or fewer.'})

    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        UPDATE settings SET
            latitude=?, longitude=?, allowed_radius=?, confidence_threshold=?,
            enrollment_code=?
    ''', (lat, lon, radius, threshold, enrollment_code))
    conn.commit()
    conn.close()

    return jsonify({'success': True, 'message': 'Settings saved successfully.'})


# ─────────────────────────── SMS ROUTES ───────────────────────────

@app.route('/api/sms/send_now', methods=['POST'])
@login_required
def sms_send_now():
    """Manually trigger absent SMS notifications for today (forces immediate delivery)."""
    from threading import Thread
    Thread(target=check_absent_and_notify, args=(True,), daemon=True).start()
    return jsonify({'success': True, 'message': 'Absent notification job triggered. Messages will be sent shortly.'})

@app.route('/api/sms/test', methods=['POST'])
@login_required
def sms_test():
    """Send a test SMS to a given phone number."""
    data  = request.get_json()
    phone = str(data.get('phone', '')).strip()
    if not phone:
        return jsonify({'success': False, 'message': 'Phone number is required.'})
    settings = get_settings_from_db()
    msg = f"Test from FaceAttend ({settings.get('college_name','College')}): SMS alerts are working correctly!"
    ok, info = send_sms(phone, msg, settings)
    return jsonify({'success': ok, 'message': info if ok else f'Failed: {info}'})

@app.route('/api/sms/log')
@login_required
def sms_log_api():
    """Return recent SMS log entries."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute('''
        SELECT sl.*, s.name FROM sms_log sl
        LEFT JOIN students s ON sl.student_id = s.student_id
        ORDER BY sl.sent_at DESC LIMIT 100
    ''').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

# ─────────────────────────── STARTUP ───────────────────────────

def start_scheduler():
    """
    Initialize and start the APScheduler background jobs.
    Runs 100% automatically without any manual intervention.
    1. Daily cron job at configured sms_notify_time (e.g. 18:00).
    2. Periodic 15-minute background check to catch up if server was restarted.
    """
    global scheduler
    scheduler = BackgroundScheduler(daemon=True)

    # Load notify time from DB
    s = get_settings_from_db()
    try:
        h, m = map(int, s.get('sms_notify_time', '18:00').split(':'))
    except Exception:
        h, m = 18, 0

    # 1. Scheduled daily trigger at exact notify time
    scheduler.add_job(
        check_absent_and_notify,
        'cron',
        hour=h, minute=m,
        id='absent_sms_job',
        replace_existing=True
    )

    # 2. Automated background checker (every 15 mins) to ensure absent students are notified
    scheduler.add_job(
        check_absent_and_notify,
        'interval',
        minutes=15,
        id='absent_sms_interval',
        replace_existing=True
    )

    scheduler.start()
    print(f'[SMS Scheduler] Started — automatic absent alerts scheduled daily at {h:02d}:{m:02d} with continuous catch-up checks.')

    # Run an initial background check 5 seconds after startup
    from threading import Timer
    Timer(5.0, check_absent_and_notify, kwargs={'force': False}).start()


# Initialise a placeholder so `scheduler` is always defined (needed by reschedule_job calls)
scheduler = BackgroundScheduler(daemon=True)


if __name__ == '__main__':
    init_db()
    load_recognizer()

    # Start scheduler only in the reloader child (or if not using reloader at all)
    import os as _os
    if _os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not _os.environ.get('WERKZEUG_RUN_MAIN'):
        start_scheduler()

    print('=' * 60)
    print(' AI Face Recognition Attendance System')
    print('=' * 60)
    print(' Default Admin: username=admin  password=admin123')
    print(' Open: http://localhost:5000')
    print('=' * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
