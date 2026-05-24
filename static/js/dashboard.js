/**
 * dashboard.js — Logic for fetching and rendering Model Training History
 */

document.addEventListener("DOMContentLoaded", () => {
    fetchTrainingHistory();
});

function fetchTrainingHistory() {
    fetch("/api/training-history")
        .then(res => res.json())
        .then(data => {
            if (data.available) {
                renderDashboard(data);
            } else {
                alert("Training history is not available. Please ensure the model has been trained and outputs exist.");
            }
        })
        .catch(err => {
            console.error("Failed to fetch training history:", err);
            alert("Failed to load dashboard data. See console for details.");
        });
}

function renderDashboard(data) {
    // 1. Update Summary Cards
    const testAccEl = document.getElementById("summary-test-acc");
    const testLossEl = document.getElementById("summary-test-loss");
    const epochsEl = document.getElementById("summary-epochs");

    if (data.test_accuracy !== undefined) {
        testAccEl.textContent = (data.test_accuracy * 100).toFixed(2) + "%";
    }
    if (data.test_loss !== undefined) {
        testLossEl.textContent = data.test_loss.toFixed(4);
    }
    
    const epochs = data.accuracy ? data.accuracy.length : 0;
    epochsEl.textContent = epochs;

    // Generate Labels (Epoch 1, Epoch 2, ...)
    const labels = Array.from({ length: epochs }, (_, i) => `Epoch ${i + 1}`);

    // Chart Global Defaults
    Chart.defaults.color = "#9ea2c0"; // var(--text-secondary)
    Chart.defaults.font.family = "'Inter', sans-serif";

    // 2. Render Accuracy Chart
    const ctxAcc = document.getElementById('accuracyChart').getContext('2d');
    new Chart(ctxAcc, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Training Accuracy',
                    data: data.accuracy,
                    borderColor: '#00f2fe', // var(--accent-cyan)
                    backgroundColor: 'rgba(0, 242, 254, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Validation Accuracy',
                    data: data.val_accuracy,
                    borderColor: '#8a2be2', // var(--accent-purple)
                    backgroundColor: 'rgba(138, 43, 226, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.4,
                    fill: false
                }
            ]
        },
        options: getChartOptions('Accuracy')
    });

    // 3. Render Loss Chart
    const ctxLoss = document.getElementById('lossChart').getContext('2d');
    new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Training Loss',
                    data: data.loss,
                    borderColor: '#ff007f', // var(--accent-pink)
                    backgroundColor: 'rgba(255, 0, 127, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Validation Loss',
                    data: data.val_loss,
                    borderColor: '#f1f3f9', // var(--text-primary)
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.4,
                    fill: false
                }
            ]
        },
        options: getChartOptions('Loss')
    });
}

function getChartOptions(title) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#f1f3f9',
                    font: {
                        size: 13
                    }
                }
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(22, 24, 47, 0.9)',
                titleColor: '#00f2fe',
                bodyColor: '#f1f3f9',
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1
            }
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                    drawBorder: false
                }
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                    drawBorder: false
                }
            }
        },
        interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
        }
    };
}
