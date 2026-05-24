/**
 * script.js — Frontend Interaction Logic
 * ===================================================
 * Manages drag-and-drop file upload, preview, network requests,
 * dynamic progress bar animations, confidence circle rendering,
 * and error handling.
 */

document.addEventListener("DOMContentLoaded", () => {
    // ──────────────────────────────────────────────────────────────────
    // Elements
    // ──────────────────────────────────────────────────────────────────
    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const previewSection = document.getElementById("preview-section");
    const previewImage = document.getElementById("preview-image");
    const removeBtn = document.getElementById("remove-btn");
    const fileNameText = document.getElementById("file-name");
    const predictBtn = document.getElementById("predict-btn");
    
    const resultsPlaceholder = document.getElementById("results-placeholder");
    const resultsSection = document.getElementById("results-section");
    const predictedClassText = document.getElementById("predicted-class");
    const confidenceValueText = document.getElementById("confidence-value");
    const confidenceRing = document.getElementById("confidence-ring");
    const predictionsList = document.getElementById("predictions-list");
    
    const resultImageContainer = document.getElementById("result-image-container");
    const resultImage = document.getElementById("result-image");
    
    const loadingOverlay = document.getElementById("loading-overlay");
    
    // Graph elements
    const accuracyGraph = document.getElementById("accuracy-graph");
    const lossGraph = document.getElementById("loss-graph");
    const confusionGraph = document.getElementById("confusion-graph");

    let selectedFile = null;
    const CIRCUMFERENCE = 2 * Math.PI * 58; // 364.42

    // ──────────────────────────────────────────────────────────────────
    // Initialization
    // ──────────────────────────────────────────────────────────────────
    resetPredictions();
    setupGraphFallbacks();

    // ──────────────────────────────────────────────────────────────────
    // Graph Fallback Handling
    // ──────────────────────────────────────────────────────────────────
    function setupGraphFallbacks() {
        [accuracyGraph, lossGraph, confusionGraph].forEach(img => {
            if (!img) return;
            
            // Check if already complete but failed
            if (img.complete && img.naturalWidth === 0) {
                handleGraphError(img);
            }
            
            img.addEventListener("error", () => {
                handleGraphError(img);
            });
        });
    }

    function handleGraphError(imgElement) {
        imgElement.style.display = "none";
        const wrapper = imgElement.closest(".metric-image-wrapper");
        if (wrapper) {
            const fallback = wrapper.querySelector(".metric-fallback");
            if (fallback) {
                fallback.style.display = "flex";
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // Drag and Drop events
    // ──────────────────────────────────────────────────────────────────
    uploadZone.addEventListener("click", () => fileInput.click());
    
    // Support keyboard activation
    uploadZone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInput.click();
        }
    });

    ["dragenter", "dragover"].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.remove("dragover");
        }, false);
    });

    uploadZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        clearFileSelection();
    });

    // ──────────────────────────────────────────────────────────────────
    // File Management Functions
    // ──────────────────────────────────────────────────────────────────
    function handleFileSelection(file) {
        if (!file.type.startsWith("image/")) {
            showError("Selected file must be an image.");
            return;
        }

        selectedFile = file;
        fileNameText.textContent = file.name;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewSection.style.display = "flex";
            uploadZone.style.display = "none";
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearFileSelection() {
        selectedFile = null;
        fileInput.value = "";
        previewImage.src = "";
        previewSection.style.display = "none";
        uploadZone.style.display = "flex";
        predictBtn.disabled = true;
        
        resetPredictions();
    }

    // ──────────────────────────────────────────────────────────────────
    // Prediction Handling
    // ──────────────────────────────────────────────────────────────────
    predictBtn.addEventListener("click", () => {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append("file", selectedFile);

        // Show spinner / loading overlay
        showLoading(true);

        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => {
                    throw new Error(errData.error || `HTTP error! Status: ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            showLoading(false);
            if (data.success) {
                displayResults(data);
            } else {
                showError(data.error || "An unknown classification error occurred.");
            }
        })
        .catch(err => {
            showLoading(false);
            showError(err.message || "Failed to communicate with Flask backend API.");
        });
    });

    function showLoading(isLoading) {
        if (isLoading) {
            loadingOverlay.classList.add("active");
            predictBtn.disabled = true;
        } else {
            loadingOverlay.classList.remove("active");
            predictBtn.disabled = false;
        }
    }

    function resetPredictions() {
        resultsPlaceholder.style.display = "flex";
        resultsSection.style.display = "none";
        resultImageContainer.style.display = "none";
        resultImage.src = "";
        
        // Reset confidence ring
        confidenceRing.style.strokeDashoffset = CIRCUMFERENCE;
        confidenceValueText.textContent = "0%";
        predictedClassText.textContent = "—";
    }

    function displayResults(data) {
        resultsPlaceholder.style.display = "none";
        resultsSection.style.display = "flex";

        const topPred = data.predictions[0];
        
        // Update top prediction text
        predictedClassText.textContent = topPred.class;
        
        // Animate circular confidence ring
        animateConfidenceRing(topPred.confidence);

        // Populate bottom predictions list
        predictionsList.innerHTML = "";
        
        data.predictions.forEach((pred, index) => {
            const card = document.createElement("div");
            card.className = `prediction-card prediction-card--rank-${index + 1}`;
            
            card.innerHTML = `
                <div class="prediction-info">
                    <span class="prediction-class-name">
                        <span class="prediction-rank">#${index + 1}</span>
                        ${pred.class}
                    </span>
                    <span class="prediction-percentage">${pred.confidence.toFixed(1)}%</span>
                </div>
                <div class="progress-bar-outer">
                    <div class="progress-bar-fill" id="bar-fill-${index}"></div>
                </div>
            `;
            
            predictionsList.appendChild(card);
            
            // Trigger animation in next tick
            setTimeout(() => {
                const fill = document.getElementById(`bar-fill-${index}`);
                if (fill) {
                    fill.style.width = `${pred.confidence}%`;
                }
            }, 50);
        });

        // Set prediction thumbnail if provided
        if (data.image_url) {
            resultImage.src = data.image_url;
            resultImageContainer.style.display = "flex";
        } else {
            resultImageContainer.style.display = "none";
        }
    }

    function animateConfidenceRing(percentage) {
        let currentPercent = 0;
        const targetPercent = percentage;
        const duration = 800; // matching transition duration in css
        const startTime = performance.now();

        function update(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Ease out cubic
            const ease = 1 - Math.pow(1 - progress, 3);
            currentPercent = ease * targetPercent;
            
            // Update dash offset
            const offset = CIRCUMFERENCE - (currentPercent / 100) * CIRCUMFERENCE;
            confidenceRing.style.strokeDashoffset = offset;
            
            // Update text
            confidenceValueText.textContent = `${Math.round(currentPercent)}%`;

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                // Ensure exact final value is shown
                confidenceRing.style.strokeDashoffset = CIRCUMFERENCE - (targetPercent / 100) * CIRCUMFERENCE;
                confidenceValueText.textContent = `${targetPercent.toFixed(1)}%`;
            }
        }

        requestAnimationFrame(update);
    }

    // ──────────────────────────────────────────────────────────────────
    // Error notification
    // ──────────────────────────────────────────────────────────────────
    function showError(message) {
        alert(`❌ Error:\n\n${message}`);
    }
});
