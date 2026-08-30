// Student Info Management
function loadStudentInfo() {
    const defaultId = "67114540116";
    const defaultFirst = "ชานนท์";
    const defaultLast = "สายแจ้";

    const savedId = localStorage.getItem('dl69_student_id') || defaultId;
    const savedFirst = localStorage.getItem('dl69_student_first') || defaultFirst;
    const savedLast = localStorage.getItem('dl69_student_last') || defaultLast;

    document.getElementById('studentId').textContent = savedId;
    document.getElementById('studentFirstName').textContent = savedFirst;
    document.getElementById('studentLastName').textContent = savedLast;
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

// รับจุดข้อมูลจาก Backend PyTorch
let pts = [];

function draw(s) {
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, 400, 400);

    // วาดแกน Grid อ่อนๆ
    ctx.strokeStyle = '#27273f';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(CX, 0); ctx.lineTo(CX, 400);
    ctx.moveTo(0, CX); ctx.lineTo(400, CX);
    ctx.stroke();

    // วาดจุดข้อมูลจริงที่สร้างจาก PyTorch (Class 1 = เขียว, Class 0 = แดง)
    for (const p of pts) {
        ctx.beginPath();
        ctx.arc(CX + p.x * S, CX - p.y * S, 3.5, 0, 6.283);
        ctx.fillStyle = p.l ? '#4ade80' : '#f87171';
        ctx.fill();
    }

    // วาดเส้นแบ่ง Decision Boundary: w0*x + w1*y + b = 0 => y = -(w0*x + b)/w1
    if (s && s.w && s.w[1] !== 0) {
        ctx.beginPath();
        const y1 = -(s.w[0] * (-3.5) + s.b) / s.w[1];
        const y2 = -(s.w[0] * (3.5) + s.b) / s.w[1];
        ctx.moveTo(0, CX - y1 * S);
        ctx.lineTo(400, CX - y2 * S);
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2.5;
        ctx.stroke();
    }

    // อัปเดตตัวเลขสถานะแบบ Realtime
    if (s && s.epoch !== undefined) {
        if (document.getElementById('metricEpoch')) {
            document.getElementById('metricEpoch').textContent = `${s.epoch} / ${s.total_epochs || 200}`;
            document.getElementById('metricLoss').textContent = s.loss.toFixed(4);
            document.getElementById('metricAcc').textContent = (s.acc * 100).toFixed(1) + '%';
            if (s.w) {
                document.getElementById('metricParams').textContent = `w: [${s.w[0]}, ${s.w[1]}] | b: ${s.b}`;
            }
        }

        if (document.getElementById('info')) {
            document.getElementById('info').textContent =
                `Epoch ${s.epoch} / ${s.total_epochs || 200} | Loss: ${s.loss.toFixed(4)} | Accuracy: ${(s.acc * 100).toFixed(1)}%`;
        }
    }
}

// EventSource stream management
let es = null;

function connectSSE() {
    if (es) {
        es.close();
    }

    const badge = document.getElementById('statusBadge');
    if (badge) {
        badge.textContent = '⚡ กำลังเทรนโมเดลสด...';
        badge.style.borderColor = '#3b82f6';
        badge.style.color = '#60a5fa';
    }

    es = new EventSource('/api/train/');
    
    es.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.type === 'init') {
            pts = data.points || [];
            draw({ epoch: 0, loss: 0, acc: 0, w: [0, 0], b: 0 });
        } else if (data.type === 'epoch' || data.epoch !== undefined) {
            draw(data);
        }
    };

    es.onerror = () => {
        es.close();
        if (badge) {
            badge.textContent = '✓ การฝึกสอนเสร็จสมบูรณ์ (Completed)';
            badge.style.borderColor = '#10b981';
            badge.style.color = '#34d399';
        }
    };
}

function restartTraining() {
    connectSSE();
}

document.addEventListener('DOMContentLoaded', () => {
    loadStudentInfo();
    connectSSE();
});
