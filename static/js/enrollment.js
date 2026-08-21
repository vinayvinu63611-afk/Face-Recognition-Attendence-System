/**
 * enrollment.js - Handles webcam capture and student enrollment
 * for the FaceAttend AI Attendance System
 */

let stream = null;
let capturedImages = [];
const MAX_IMAGES = 10;
const MIN_IMAGES = 5;

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const noCamera = document.getElementById('noCamera');
const startCamBtn = document.getElementById('startCam');
const stopCamBtn = document.getElementById('stopCam');
const captureBtn = document.getElementById('captureBtn');
const clearBtn = document.getElementById('clearBtn');
const enrollBtn = document.getElementById('enrollBtn');
const captureCount = document.getElementById('captureCount');
const captureCount2 = document.getElementById('captureCount2');
const captureProgress = document.getElementById('captureProgress');
const thumbnailStrip = document.getElementById('thumbnailStrip');
const resultAlert = document.getElementById('resultAlert');

// Start camera
startCamBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
        video.srcObject = stream;
        video.style.display = 'block';
        noCamera.style.display = 'none';
        startCamBtn.disabled = true;
        stopCamBtn.disabled = false;
        captureBtn.disabled = false;
    } catch (err) {
        showAlert('danger', 'Camera Error: ' + (err.message || 'Cannot access camera. Please allow camera permission and try again.'));
    }
});

// Stop camera
stopCamBtn.addEventListener('click', stopCamera);

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
    video.srcObject = null;
    video.style.display = 'none';
    noCamera.style.display = 'flex';
    startCamBtn.disabled = false;
    stopCamBtn.disabled = true;
    captureBtn.disabled = true;
}

// Capture face image
captureBtn.addEventListener('click', () => {
    if (!stream) return;
    if (capturedImages.length >= MAX_IMAGES) {
        showAlert('warning', `Maximum ${MAX_IMAGES} images captured. Clear some to recapture.`);
        return;
    }

    // Draw video frame to a hidden canvas
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

    capturedImages.push(dataUrl);
    updateCaptureUI();
    addThumbnail(dataUrl, capturedImages.length - 1);

    // Flash effect on overlay
    flashOverlay();
});

function flashOverlay() {
    overlay.style.background = 'rgba(255,255,255,0.6)';
    setTimeout(() => { overlay.style.background = 'transparent'; }, 150);
}

function addThumbnail(dataUrl, index) {
    const div = document.createElement('div');
    div.className = 'position-relative';
    div.id = `thumb-${index}`;
    div.innerHTML = `
        <img src="${dataUrl}" style="width:70px;height:55px;object-fit:cover;border-radius:8px;border:2px solid #6C63FF;" />
        <span class="position-absolute top-0 start-0 badge bg-primary" style="font-size:9px;">${index+1}</span>
        <button type="button" onclick="removeImage(${index})" 
                class="position-absolute top-0 end-0 btn btn-danger btn-sm p-0" style="width:16px;height:16px;font-size:9px;border-radius:50%;">
            &times;
        </button>`;
    thumbnailStrip.appendChild(div);
}

window.removeImage = function(index) {
    capturedImages.splice(index, 1);
    thumbnailStrip.innerHTML = '';
    capturedImages.forEach((img, i) => addThumbnail(img, i));
    updateCaptureUI();
};

function updateCaptureUI() {
    const count = capturedImages.length;
    captureCount.textContent = count;
    captureCount2.textContent = `${count}/${MAX_IMAGES}`;
    const pct = (count / MAX_IMAGES) * 100;
    captureProgress.style.width = pct + '%';
    enrollBtn.disabled = count < MIN_IMAGES;
    enrollBtn.querySelector('span').textContent = count;
    clearBtn.disabled = count === 0;
}

// Clear all
clearBtn.addEventListener('click', () => {
    capturedImages = [];
    thumbnailStrip.innerHTML = '';
    updateCaptureUI();
});

// Enroll student
enrollBtn.addEventListener('click', async () => {
    const name        = document.getElementById('name').value.trim();
    const studentId   = document.getElementById('student_id').value.trim();
    const department  = document.getElementById('department').value;
    const year        = document.getElementById('year').value;
    const section     = document.getElementById('section').value;
    const email       = document.getElementById('email').value.trim();
    const phoneRaw    = document.getElementById('phone_number')?.value.trim() || '';
    const phoneNumber = phoneRaw ? '+91' + phoneRaw : '';

    if (!name || !studentId || !department || !year || !section) {
        showAlert('danger', 'Please fill in all required fields.');
        return;
    }
    if (phoneRaw && !/^\d{10}$/.test(phoneRaw)) {
        showAlert('danger', 'Phone number must be exactly 10 digits.');
        return;
    }
    if (capturedImages.length < MIN_IMAGES) {
        showAlert('danger', `Please capture at least ${MIN_IMAGES} face images.`);
        return;
    }

    enrollBtn.disabled = true;
    enrollBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Enrolling...';

    try {
        const response = await fetch('/api/enroll', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name, student_id: studentId, department, year, section,
                email, phone_number: phoneNumber,
                images: capturedImages
            })
        });
        const data = await response.json();

        if (data.success) {
            showAlert('success', `<i class="bi bi-check-circle-fill me-2"></i><strong>Face Enrollment Successful!</strong><br>${data.message}`);
            // Reset form
            document.getElementById('enrollmentForm').reset();
            capturedImages = [];
            thumbnailStrip.innerHTML = '';
            updateCaptureUI();
        } else {
            showAlert('danger', `<i class="bi bi-exclamation-triangle-fill me-2"></i>${data.message}`);
        }
    } catch (err) {
        showAlert('danger', 'Network error. Please check your connection and try again.');
    } finally {
        enrollBtn.disabled = capturedImages.length < MIN_IMAGES;
        enrollBtn.innerHTML = `<i class="bi bi-check-circle-fill me-2"></i>Enroll Student (<span id="captureCount">${capturedImages.length}</span>/5)`;
    }
});

function showAlert(type, message) {
    resultAlert.style.display = 'block';
    resultAlert.className = `alert alert-${type}`;
    resultAlert.innerHTML = message;
    resultAlert.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Initialize
video.style.display = 'none';
updateCaptureUI();
