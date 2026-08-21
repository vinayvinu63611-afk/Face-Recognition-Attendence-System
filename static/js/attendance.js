/**
 * attendance.js - Handles live webcam recognition and attendance marking
 * for the FaceAttend AI Attendance System
 */

let stream = null;
let recognitionInterval = null;
let currentLocation = null;
let lastRecognitionData = null;
let isProcessing = false;

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const noCamera = document.getElementById('noCamera');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const markBtn = document.getElementById('markBtn');
const statusBadge = document.getElementById('statusBadge');
const locationBar = document.getElementById('locationBar');
const resultCard = document.getElementById('resultCard');
const attendanceAlert = document.getElementById('attendanceAlert');
const userCoordsEl = document.getElementById('userCoords');
const distanceEl = document.getElementById('distanceInfo');
const locationStatusEl = document.getElementById('locationStatus');
const todayCount = document.getElementById('todayCount');
const todayDate = document.getElementById('todayDate');

// Set today's date
todayDate.textContent = new Date().toLocaleDateString('en-IN', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });

// Load today's count
async function loadTodayCount() {
    try {
        const res = await fetch('/api/dashboard_stats');
        const data = await res.json();
        todayCount.textContent = data.present_today;
    } catch(e) {}
}
loadTodayCount();

// ── FAST GEOLOCATION ENGINE ──────────────────────────────────
let locationWatchId = null;
let locationPromise = null;

function applyPosition(pos) {
    currentLocation = {
        lat: pos.coords.latitude,
        lon: pos.coords.longitude,
        accuracy: pos.coords.accuracy
    };

    const acc = pos.coords.accuracy;
    const accLabel = acc < 50 ? `<span class="badge bg-success">${acc.toFixed(0)} m ✓</span>` :
                     acc < 200 ? `<span class="badge bg-warning text-dark">${acc.toFixed(0)} m</span>` :
                                 `<span class="badge bg-danger">${acc.toFixed(0)} m</span>`;

    const coordsEl = document.getElementById('userCoords');
    const accEl = document.getElementById('gpsAccuracy');
    const locationBar = document.getElementById('locationBar');
    const panelAlert = document.getElementById('locationPanelAlert');
    const refreshBtn = document.getElementById('refreshLocationBtn');

    if (coordsEl) coordsEl.textContent = `${pos.coords.latitude.toFixed(5)}, ${pos.coords.longitude.toFixed(5)}`;
    if (accEl) accEl.innerHTML = accLabel;
    if (panelAlert) panelAlert.style.display = 'none';
    if (refreshBtn) { refreshBtn.disabled = false; refreshBtn.innerHTML = '<i class="bi bi-arrow-repeat me-1"></i>Refresh'; }

    if (locationBar) {
        locationBar.className = 'alert alert-success d-flex align-items-center mb-3';
        locationBar.innerHTML = `<i class="bi bi-geo-alt-fill me-2"></i><small><strong>Location ready</strong> — (${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)})</small>`;
        setTimeout(() => { locationBar.style.display = 'none'; }, 2500);
    }
}

// Ultra-fast multi-tier location acquisition
function getLocation(forceFresh = false) {
    if (currentLocation && !forceFresh) {
        return Promise.resolve(currentLocation);
    }

    if (!navigator.geolocation) {
        return Promise.resolve(null);
    }

    const locationBar = document.getElementById('locationBar');
    const refreshBtn = document.getElementById('refreshLocationBtn');
    if (locationBar) {
        locationBar.style.display = 'flex';
        locationBar.className = 'alert alert-info d-flex align-items-center mb-3';
        locationBar.innerHTML = '<div class="spinner-border spinner-border-sm me-2" role="status"></div><small>Detecting GPS location...</small>';
    }
    if (refreshBtn) { refreshBtn.disabled = true; refreshBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Locating...'; }

    return new Promise((resolve) => {
        let resolved = false;

        // Tier 1: Fast cached / network position (<200ms)
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                applyPosition(pos);
                if (!resolved) { resolved = true; resolve(currentLocation); }
            },
            () => {
                // If fast lookup times out, wait for high accuracy in tier 2
            },
            { enableHighAccuracy: false, timeout: 2500, maximumAge: forceFresh ? 0 : 300000 }
        );

        // Tier 2: High-accuracy GPS refinement
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                applyPosition(pos);
                if (!resolved) { resolved = true; resolve(currentLocation); }
            },
            (err) => {
                if (!resolved) {
                    resolved = true;
                    if (refreshBtn) { refreshBtn.disabled = false; refreshBtn.innerHTML = '<i class="bi bi-arrow-repeat me-1"></i>Refresh'; }
                    resolve(currentLocation || null);
                }
            },
            { enableHighAccuracy: true, timeout: 6000, maximumAge: forceFresh ? 0 : 60000 }
        );

        // Safety timeout — never block longer than 3 seconds
        setTimeout(() => {
            if (!resolved) {
                resolved = true;
                if (refreshBtn) { refreshBtn.disabled = false; refreshBtn.innerHTML = '<i class="bi bi-arrow-repeat me-1"></i>Refresh'; }
                resolve(currentLocation || null);
            }
        }, 3000);
    });
}

// Start background continuous watcher for instant real-time coordinates
function startBackgroundLocationWatcher() {
    if (!navigator.geolocation || locationWatchId != null) return;
    try {
        locationWatchId = navigator.geolocation.watchPosition(
            (pos) => applyPosition(pos),
            (err) => {},
            { enableHighAccuracy: true, maximumAge: 30000, timeout: 10000 }
        );
    } catch(e) {}
}

// Pre-fetch location immediately when page loads!
getLocation();
startBackgroundLocationWatcher();

// Standalone refresh button handler
async function refreshLocation() {
    currentLocation = null;
    document.getElementById('userCoords').textContent = '---';
    document.getElementById('gpsAccuracy').textContent = '---';
    document.getElementById('distanceInfo').textContent = '---';
    document.getElementById('locationStatus').textContent = '---';
    await getLocation(true);
}

// Start recognition — ultra-fast parallel launch
startBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
        video.srcObject = stream;
        video.style.display = 'block';
        noCamera.style.display = 'none';
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusBadge.className = 'badge bg-success';
        statusBadge.textContent = 'Camera On';

        // Trigger non-blocking location refresh if not ready
        if (!currentLocation) {
            getLocation();
        }

        // Start auto-recognition every 1.5 seconds (fast response)
        if (recognitionInterval) clearInterval(recognitionInterval);
        recognitionInterval = setInterval(autoRecognize, 1500);

        // Immediate first scan after 300ms warm-up
        setTimeout(autoRecognize, 300);
    } catch (err) {
        showAttendanceAlert('danger', 'Camera Error: ' + (err.message || 'Cannot access camera.'));
    }
});

// Stop recognition
stopBtn.addEventListener('click', stopRecognition);

function stopRecognition() {
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (recognitionInterval) { clearInterval(recognitionInterval); recognitionInterval = null; }
    video.srcObject = null;
    video.style.display = 'none';
    noCamera.style.display = 'flex';
    startBtn.disabled = false;
    stopBtn.disabled = true;
    markBtn.disabled = true;
    statusBadge.className = 'badge bg-secondary';
    statusBadge.textContent = 'Camera Off';
    lastRecognitionData = null;
}

// Auto-recognize face from webcam
async function autoRecognize() {
    if (isProcessing || !stream) return;
    isProcessing = true;
    statusBadge.className = 'badge bg-warning text-dark';
    statusBadge.textContent = 'Scanning...';

    try {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg', 0.8);

        const payload = {
            image: imageData,
            latitude: currentLocation?.lat || null,
            longitude: currentLocation?.lon || null
        };

        const response = await fetch('/api/recognize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        handleRecognitionResult(data);
    } catch (err) {
        console.error('Recognition error:', err);
    } finally {
        isProcessing = false;
        if (stream) {
            statusBadge.className = 'badge bg-success';
            statusBadge.textContent = 'Camera On';
        }
    }
}

function handleRecognitionResult(data) {
    if (data.success) {
        // Attendance marked!
        lastRecognitionData = data;
        markBtn.disabled = false;
        showRecognitionCard(data, true);
        showAttendanceAlert('success', `
            <i class="bi bi-check-circle-fill me-2"></i>
            <strong>Attendance Marked Successfully!</strong><br>
            <small>${data.message}</small>`);
        updateLocationUI(data);
        loadTodayCount();
        // Stop auto-recognition after success
        if (recognitionInterval) { clearInterval(recognitionInterval); recognitionInterval = null; }
    } else if (data.already_marked) {
        lastRecognitionData = data;
        showRecognitionCard(data, false);
        showAttendanceAlert('warning', `<i class="bi bi-info-circle-fill me-2"></i>${data.message}`);
        if (recognitionInterval) { clearInterval(recognitionInterval); recognitionInterval = null; }
    } else if (data.student) {
        // Recognized but location failed
        lastRecognitionData = data;
        showRecognitionCard(data, false);
        showAttendanceAlert('danger', `<i class="bi bi-geo-alt-fill me-2"></i>${data.message}`);
        updateLocationUI(data);
    } else {
        // Not recognized or no face
        resultCard.innerHTML = `
            <div class="text-center py-2">
                <i class="bi bi-question-circle-fill text-muted" style="font-size:3rem;"></i>
                <p class="text-muted mt-2 small">${data.message}</p>
            </div>`;
    }
}

function showRecognitionCard(data, success) {
    const student = data.student;
    const icon = success ? 'bi-person-check-fill text-success' : 'bi-person-x-fill text-danger';
    resultCard.innerHTML = `
        <div class="text-start">
            <div class="d-flex align-items-center gap-3 mb-3">
                <div class="text-center">
                    <i class="bi ${icon}" style="font-size:3rem;"></i>
                </div>
                <div>
                    <h5 class="fw-bold mb-0">${student?.name || 'Unknown'}</h5>
                    <div class="text-muted small">${student?.student_id || ''}</div>
                </div>
            </div>
            <table class="table table-sm table-borderless small mb-0">
                <tr><td class="text-muted fw-semibold">Department</td><td>${student?.department || '---'}</td></tr>
                <tr><td class="text-muted fw-semibold">Year</td><td>${student?.year || '---'}</td></tr>
                <tr><td class="text-muted fw-semibold">Section</td><td>${student?.section || '---'}</td></tr>
                <tr><td class="text-muted fw-semibold">Confidence</td>
                    <td><span class="badge bg-primary">${data.confidence || '---'}%</span></td></tr>
                <tr><td class="text-muted fw-semibold">Date</td><td>${data.date || new Date().toISOString().slice(0,10)}</td></tr>
                <tr><td class="text-muted fw-semibold">Time</td><td>${data.time || '---'}</td></tr>
            </table>
        </div>`;
}

function updateLocationUI(data) {
    if (data.distance != null) {
        distanceEl.textContent = data.distance + ' m';
    }
    if (data.location_status) {
        const badge = data.location_status === 'Within Range'
            ? '<span class="badge bg-success">' + data.location_status + '</span>'
            : '<span class="badge bg-danger">' + data.location_status + '</span>';
        locationStatusEl.innerHTML = badge;
    }
}

// Manual mark attendance button (for when auto-mark needs confirmation)
markBtn.addEventListener('click', async () => {
    if (!lastRecognitionData) return;
    markBtn.disabled = true;
    await autoRecognize();
    markBtn.disabled = false;
});

function showAttendanceAlert(type, message) {
    attendanceAlert.style.display = 'block';
    attendanceAlert.className = `alert alert-${type} mb-3`;
    attendanceAlert.innerHTML = message;
}

// Initialize
video.style.display = 'none';
