const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const loadingOverlay = document.getElementById('loading');
const resultSection = document.getElementById('result-section');
const resultImage = document.getElementById('result-image');
const detectionList = document.getElementById('detection-list');

// Drag & Drop Handlers
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('이미지 파일만 업로드 가능합니다.');
        return;
    }

    uploadImage(file);
}

async function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    showLoading(true);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('서버 오류가 발생했습니다.');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        alert('이미지 분석 중 오류가 발생했습니다: ' + error.message);
    } finally {
        showLoading(false);
    }
}

function displayResults(data) {
    // Hide upload, show results
    dropZone.style.display = 'none';
    resultSection.style.display = 'flex';

    // Scroll to results smoothly
    resultSection.scrollIntoView({ behavior: 'smooth' });

    // Set image
    resultImage.src = data.image;

    // List detections
    detectionList.innerHTML = '';
    if (data.detections && data.detections.length > 0) {
        data.detections.forEach((det, index) => {
            const li = document.createElement('li');
            // Capitalize first letter
            const className = det.class_name.charAt(0).toUpperCase() + det.class_name.slice(1);
            const confidence = (det.conf * 100).toFixed(1);

            // Determine color based on confidence (optional visual cue)
            let confClass = 'high';
            if (det.conf < 0.7) confClass = 'medium';
            if (det.conf < 0.5) confClass = 'low';

            li.innerHTML = `
                <div class="defect-info">
                    <span class="defect-name">${className}</span>
                    <span class="defect-id">Defect #${index + 1}</span>
                </div>
                <div class="defect-score">
                    <span class="score-label">Confidence</span>
                    <span class="score-value ${confClass}">${confidence}%</span>
                </div>
            `;
            detectionList.appendChild(li);
        });
    } else {
        detectionList.innerHTML = '<li class="no-defect">No defects detected. PCB is clean.</li>';
    }
}

function resetApp() {
    resultSection.style.display = 'none';
    dropZone.style.display = 'block';
    fileInput.value = '';
    resultImage.src = '';
    detectionList.innerHTML = '';
}

function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}
