// --- Tab Switching Logic ---
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Remove active class from all
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.add('hidden'));
        
        // Add active to clicked
        btn.classList.add('active');
        const targetId = btn.getAttribute('data-target');
        document.getElementById(targetId).classList.remove('hidden');
    });
});

// --- 1. Symptom Prediction ---
async function predictSymptom() {
    const input = document.getElementById('symptom-input').value;
    if (!input.trim()) return alert("Please enter some symptoms.");

    // Split by comma and clean
    const symptomsList = input.split(',').map(s => s.trim()).filter(s => s);

    try {
        const response = await fetch('/api/predict/symptom', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symptoms: symptomsList })
        });
        
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        const listContainer = document.getElementById('symptom-list');
        listContainer.innerHTML = '';
        
        data.predictions.forEach(p => {
            const item = document.createElement('div');
            item.className = 'prediction-item';
            item.innerHTML = `
                <span class="prediction-name">${p.disease}</span>
                <span class="prediction-conf">${p.confidence}%</span>
            `;
            listContainer.appendChild(item);
        });

        document.getElementById('symptom-result').classList.remove('hidden');
    } catch (err) {
        alert("Error: " + err.message);
    }
}

// --- 2. Tabular Prediction (Dynamic Forms) ---
let currentFeatures = [];

async function loadDiseaseFeatures() {
    const diseaseName = document.getElementById('disease-select').value;
    if (!diseaseName) return;

    try {
        const response = await fetch(`/api/tabular/features/${diseaseName}`);
        const data = await response.json();
        
        if (data.error) throw new Error(data.error);
        
        currentFeatures = data.features;
        const formContainer = document.getElementById('dynamic-form');
        formContainer.innerHTML = '';

        // Generate inputs
        currentFeatures.forEach(feature => {
            const group = document.createElement('div');
            group.className = 'input-group';
            group.innerHTML = `
                <label>${feature.replace(/_/g, ' ')}</label>
                <input type="number" step="any" id="feat-${feature}" placeholder="Enter value..." required>
            `;
            formContainer.appendChild(group);
        });

        document.getElementById('predict-tabular-btn').classList.remove('hidden');
        document.getElementById('tabular-result').classList.add('hidden');
    } catch (err) {
        alert("Error loading features: " + err.message);
    }
}

async function predictTabular() {
    const diseaseName = document.getElementById('disease-select').value;
    const featuresData = {};
    
    // Gather all data
    let missing = false;
    currentFeatures.forEach(f => {
        const val = document.getElementById(`feat-${f}`).value;
        if (val === "") missing = true;
        featuresData[f] = parseFloat(val);
    });

    if (missing) return alert("Please fill out all fields.");

    try {
        const response = await fetch('/api/predict/tabular', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                disease: diseaseName,
                features: featuresData
            })
        });
        
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        // Animate result
        const resultBox = document.getElementById('tabular-result');
        const diagText = document.getElementById('tabular-diagnosis-text');
        const fillBar = document.getElementById('tabular-confidence-fill');
        const confText = document.getElementById('tabular-confidence-text');

        resultBox.classList.remove('hidden');
        
        diagText.innerText = data.prediction_text;
        diagText.style.color = data.prediction === 1 ? 'var(--danger)' : 'var(--success)';
        
        // Reset bar for animation
        fillBar.style.width = '0%';
        
        setTimeout(() => {
            fillBar.style.width = `${data.confidence}%`;
            fillBar.style.background = data.prediction === 1 
                ? 'linear-gradient(90deg, #ef4444, #f97316)' 
                : 'linear-gradient(90deg, #10b981, #06b6d4)';
        }, 50);

        confText.innerText = `${data.confidence}%`;

    } catch (err) {
        alert("Error: " + err.message);
    }
}

// --- 3. MRI Prediction ---
function previewImage(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('mri-preview').src = e.target.result;
        document.getElementById('preview-container').classList.remove('hidden');
        document.getElementById('predict-mri-btn').classList.remove('hidden');
        document.getElementById('mri-result').classList.add('hidden');
    }
    reader.readAsDataURL(file);
}

// Drag and drop support
const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--accent-color)';
});
dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--glass-border)';
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--glass-border)';
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        document.getElementById('mri-upload').files = e.dataTransfer.files;
        previewImage({ target: document.getElementById('mri-upload') });
    }
});

async function predictMRI() {
    const fileInput = document.getElementById('mri-upload');
    if (!fileInput.files.length) return alert("Please upload an image first.");

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    try {
        const response = await fetch('/api/predict/mri', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        // Animate result
        const resultBox = document.getElementById('mri-result');
        const diagText = document.getElementById('mri-diagnosis-text');
        const fillBar = document.getElementById('mri-confidence-fill');
        const confText = document.getElementById('mri-confidence-text');

        resultBox.classList.remove('hidden');
        
        diagText.innerText = `Detected: ${data.tumor_type.toUpperCase()}`;
        
        const isHealthy = data.tumor_type.toLowerCase() === 'notumor';
        diagText.style.color = isHealthy ? 'var(--success)' : 'var(--danger)';
        
        // Reset bar for animation
        fillBar.style.width = '0%';
        
        setTimeout(() => {
            fillBar.style.width = `${data.confidence}%`;
            fillBar.style.background = isHealthy 
                ? 'linear-gradient(90deg, #10b981, #06b6d4)' 
                : 'linear-gradient(90deg, #ef4444, #f97316)';
        }, 50);

        confText.innerText = `${data.confidence}%`;

    } catch (err) {
        alert("Error: " + err.message);
    }
}
