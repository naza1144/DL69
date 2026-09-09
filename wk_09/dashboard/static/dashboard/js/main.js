const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const CX = 200, S = 60;

let pts = [];
let es = null;

function draw(s) {
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, 400, 400);

    // Draw grid
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(CX, 0); ctx.lineTo(CX, 400);
    ctx.moveTo(0, CX); ctx.lineTo(400, CX);
    ctx.stroke();

    // Draw data points (Class 1 = green, Class 0 = red)
    for (const p of pts) {
        ctx.beginPath();
        ctx.arc(CX + p.x * S, CX - p.y * S, 3.5, 0, 6.283);
        ctx.fillStyle = p.l ? '#4ade80' : '#f87171';
        ctx.fill();
    }

    // Draw decision boundary: w0*x + w1*y + b = 0 => y = -(w0*x + b)/w1
    if (s && s.w && s.w[1] !== 0) {
        ctx.beginPath();
        const y1 = -(s.w[0] * (-3.5) + s.b) / s.w[1];
        const y2 = -(s.w[0] * (3.5) + s.b) / s.w[1];
        ctx.moveTo(0, CX - y1 * S);
        ctx.lineTo(400, CX - y2 * S);
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // Update text metrics
    if (s && s.epoch !== undefined) {
        document.getElementById('valEpoch').textContent = `${s.epoch} / ${s.total_epochs || 200}`;
        document.getElementById('valLoss').textContent = s.loss.toFixed(4);
        const accPct = (s.accuracy || s.acc || 0) * (s.accuracy <= 1 ? 100 : 1);
        document.getElementById('valAcc').textContent = accPct.toFixed(1) + '%';
    }
}

function startTraining() {
    const btn = document.getElementById('btnStart');
    btn.disabled = true;

    if (es) {
        es.close();
    }

    // เปิด EventSource เชื่อมต่อไปยัง /train/
    const endpoint = '/train/';
    es = new EventSource(endpoint);

    let lastLrResults = null;

    es.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.type === 'init') {
            pts = data.points || [];
            draw({ epoch: 0, loss: 0, acc: 0, w: [0, 0], b: 0 });
        } else if (data.type === 'epoch' || data.epoch !== undefined) {
            if (data.lr_results) {
                lastLrResults = data.lr_results;
            }
            draw(data);
        }
    };

    es.onerror = () => {
        es.close();
        btn.disabled = false;
        showResults(lastLrResults);
    };
}

function showResults(lrResults) {
    const resultsDiv = document.getElementById('results');
    const tableBody = document.getElementById('lrTableBody');
    const img = document.getElementById('lossCurveImg');

    if (resultsDiv) resultsDiv.style.display = 'block';

    if (lrResults && tableBody) {
        tableBody.innerHTML = '';
        const sortedKeys = Object.keys(lrResults).sort((a, b) => parseFloat(a) - parseFloat(b));
        sortedKeys.forEach(lrKey => {
            const item = lrResults[lrKey];
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${item.lr}</strong></td>
                <td>${item.final_loss}</td>
                <td>${item.final_acc}%</td>
            `;
            tableBody.appendChild(tr);
        });
    }

    if (img) {
        // reload image cache
        img.src = '/static/dashboard/loss_curve.png?t=' + new Date().getTime();
    }
}

// Draw initial empty state
draw({ epoch: 0, loss: 0, acc: 0, w: [0, 0], b: 0 });
