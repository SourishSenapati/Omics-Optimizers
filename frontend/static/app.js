let chart = null;
const API_URL = ""; // Serve from same origin as static files

// Initialization logic
document.addEventListener("DOMContentLoaded", () => {
    initChart();
    fetchForensicIntel();
    checkHardwareStatus();
    
    // UI Event Handlers
    const slider = document.getElementById("intervention-slider");
    const sliderValue = document.getElementById("intervention-value");
    
    slider.addEventListener("input", (e) => {
        const val = e.target.value;
        sliderValue.textContent = `${val}%`;
    });

    const runBtn = document.getElementById("run-btn");
    runBtn.addEventListener("click", () => runInference());
});

async function checkHardwareStatus() {
    const hwBadge = document.getElementById("hw-badge");
    try {
        const res = await fetch(`${API_URL}/heartbeat`);
        const data = await res.json();
        hwBadge.textContent = "Engine Active | CUDA Detected";
        hwBadge.style.color = "var(--accent-teal)";
        hwBadge.style.borderColor = "var(--accent-teal)";
        document.getElementById("run-btn").innerHTML = '<i data-lucide="play"></i> Trigger PINN Inference';
        lucide.createIcons();
    } catch (e) {
        hwBadge.textContent = "Engine Offline | No Link";
        hwBadge.style.color = "var(--accent-red)";
        hwBadge.style.borderColor = "var(--accent-red)";
    }
}

async function fetchForensicIntel() {
    const intelContainer = document.getElementById("intel-container");
    try {
        const res = await fetch(`${API_URL}/forensic_feed`);
        const data = await res.json();
        
        intelContainer.innerHTML = "";
        data.forEach(item => {
            const div = document.createElement("div");
            div.className = "intel-item";
            div.innerHTML = `
                <h4>${item.title.substring(0, 50)}...</h4>
                <p>${item.summary.substring(0, 100)}...</p>
                <div class="intel-meta">
                    <span>${item.forensic_intelligence.threat_level.toUpperCase()} THREAT</span>
                    <span>${item.published}</span>
                </div>
            `;
            intelContainer.appendChild(div);
        });
    } catch (e) {
        intelContainer.innerHTML = `<div class="intel-item loading">Intelligence Link Failed.</div>`;
    }
}

function initChart() {
    const ctx = document.getElementById('mainChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Historical Data',
                    borderColor: '#2dd4bf',
                    backgroundColor: 'rgba(45, 212, 191, 0.1)',
                    borderWidth: 2,
                    data: [],
                    pointRadius: 4,
                    tension: 0.1
                },
                {
                    label: 'PINN Projection',
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.1)',
                    borderWidth: 3,
                    borderDash: [5, 5],
                    data: [],
                    pointRadius: 0,
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#8b949e', font: { family: 'Outfit' } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#8b949e', font: { family: 'Outfit' } }
                }
            }
        }
    });
}

async function runInference() {
    const runBtn = document.getElementById("run-btn");
    const intensity = document.getElementById("intervention-slider").value / 100;
    const epochs = document.getElementById("training-depth").value;

    runBtn.disabled = true;
    runBtn.innerHTML = '<i data-lucide="loader-2" class="spin"></i> Computing Gradient...';
    lucide.createIcons();

    try {
        const res = await fetch(`${API_URL}/train`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                epochs: parseInt(epochs),
                learning_rate: 0.001,
                intervention_factor: parseFloat(intensity)
            })
        });
        
        const data = await res.json();
        updateUI(data.results);
    } catch (e) {
        console.error("Inference Error:", e);
        alert("Failed to communicate with calculation engine. Ensure backend server is running.");
    } finally {
        runBtn.disabled = false;
        runBtn.innerHTML = '<i data-lucide="play"></i> Retrigger PINN Inference';
        lucide.createIcons();
    }
}

function updateUI(results) {
    // Update Stats
    const kinetics = results.metadata.kinetics;
    document.getElementById("stat-beta").textContent = kinetics.beta.toFixed(4);
    document.getElementById("stat-gamma").textContent = kinetics.gamma.toFixed(4);
    document.getElementById("stat-r0").textContent = kinetics.r0.toFixed(2);

    // Update Chart
    const labels = Array.from({length: results.historical_fit.length + results.prediction_next_7_days.length}, (_, i) => `Day ${i}`);
    
    chart.data.labels = labels;
    
    // Historical Data
    const emptyPadding = new Array(results.prediction_next_7_days.length).fill(null);
    chart.data.datasets[0].data = results.historical_fit.concat(emptyPadding);

    // Prediction Data (Joined with last point of historical for visual continuity)
    const padding = new Array(results.historical_fit.length - 1).fill(null);
    const lastPoint = results.historical_fit[results.historical_fit.length - 1];
    chart.data.datasets[1].data = padding.concat([lastPoint]).concat(results.prediction_next_7_days);

    chart.update();
}
