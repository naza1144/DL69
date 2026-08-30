const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const CX = 200, S = 60;

let pts = [];

function draw(s) {
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, 400, 400);

    for (const p of pts) {
        ctx.beginPath();
        ctx.arc(CX + p.x * S, CX - p.y * S, 3.5, 0, 6.283);
        ctx.fillStyle = p.l ? '#4ade80' : '#f87171';
        ctx.fill();
    }

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

    if (s && s.epoch !== undefined && document.getElementById('info')) {
        document.getElementById('info').textContent =
            `epoch ${s.epoch} | loss ${s.loss} | acc ${(s.acc * 100).toFixed(0)}%`;
    }
}

const es = new EventSource('/api/train/');

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
};
