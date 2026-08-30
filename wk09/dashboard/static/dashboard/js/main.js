// Student Info Management
function loadStudentInfo() {
    const savedId = localStorage.getItem('dl69_student_id');
    const savedFirst = localStorage.getItem('dl69_student_first');
    const savedLast = localStorage.getItem('dl69_student_last');
    if (savedId) document.getElementById('studentId').textContent = savedId;
    if (savedFirst) document.getElementById('studentFirstName').textContent = savedFirst;
    if (savedLast) document.getElementById('studentLastName').textContent = savedLast;
}

function editStudentInfo() {
    const curId = document.getElementById('studentId').textContent;
    const curFirst = document.getElementById('studentFirstName').textContent;
    const curLast = document.getElementById('studentLastName').textContent;

    const newId = prompt("กรอกรหัสนักศึกษา (Student ID):", curId);
    if (newId === null) return;
    const newFirst = prompt("กรอกชื่อ (First Name):", curFirst);
    if (newFirst === null) return;
    const newLast = prompt("กรอกนามสกุล (Last Name):", curLast);
    if (newLast === null) return;

    localStorage.setItem('dl69_student_id', newId);
    localStorage.setItem('dl69_student_first', newFirst);
    localStorage.setItem('dl69_student_last', newLast);
    loadStudentInfo();
}

// Canvas & Realtime Visualization
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const CX = 200, S = 60;

// ข้อมูลจำลอง (Seed เดียวกับ Backend)
const pts = [];
function randn() {
    let u = 0, v = 0;
    while (!u) u = Math.random();
    v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283 * v);
}

for (let i = 0; i < 200; i++) {
    const x = randn(), y = randn();
    pts.push({ x, y, l: (x * 1.5 + y - 0.5) > 0 ? 1 : 0 });
}

function draw(s) {
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, 400, 400);

    // วาดจุดข้อมูล (Class 1 = เขียว, Class 0 = แดง)
    for (const p of pts) {
        ctx.beginPath();
        ctx.arc(CX + p.x * S, CX - p.y * S, 3.5, 0, 6.283);
        ctx.fillStyle = p.l ? '#4ade80' : '#f87171';
        ctx.fill();
    }

    // วาด Decision Boundary Line: w0*x + w1*y + b = 0
    if (s.w && s.w[1] !== 0) {
        ctx.beginPath();
        ctx.moveTo(0, CX - (-(s.w[0] * (-3) + s.b) / s.w[1]) * S);
        ctx.lineTo(400, CX - (-(s.w[0] * (3) + s.b) / s.w[1]) * S);
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2.5;
        ctx.stroke();
    }

    // อัปเดตตัวเลขแสดงผล Realtime
    if (document.getElementById('metricEpoch')) {
        document.getElementById('metricEpoch').textContent = s.epoch;
        document.getElementById('metricLoss').textContent = s.loss.toFixed(4);
        document.getElementById('metricAcc').textContent = (s.acc * 100).toFixed(1) + '%';
        if (s.w) {
            document.getElementById('metricParams').textContent = `w: [${s.w[0]}, ${s.w[1]}] | b: ${s.b}`;
        }
    }

    // อัปเดต text รวม
    if (document.getElementById('info')) {
        document.getElementById('info').textContent =
            `epoch ${s.epoch} | loss ${s.loss} | acc ${(s.acc * 100).toFixed(0)}%`;
    }
}

// Initial Canvas Draw
draw({ epoch: 0, loss: 0, acc: 0, w: [0, 0], b: 0 });

// Server-Sent Events (SSE) stream
const es = new EventSource('/api/train/');
es.onmessage = (e) => {
    const data = JSON.parse(e.data);
    draw(data);
};

es.onerror = () => {
    es.close();
    const badge = document.getElementById('statusBadge');
    if (badge) {
        badge.textContent = '✓ Training Completed';
        badge.style.borderColor = '#10b981';
        badge.style.color = '#34d399';
    }
};

document.addEventListener('DOMContentLoaded', () => {
    loadStudentInfo();
});
