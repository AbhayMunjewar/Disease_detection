const chatBox = document.getElementById('chat-box');
const chatInput = document.getElementById('chat-input');

// Append a message bubble to the chat
function appendMessage(sender, htmlContent) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}-message`;
    
    const icon = sender === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
    
    msgDiv.innerHTML = `
        <div class="avatar">${icon}</div>
        <div class="bubble">${htmlContent}</div>
    `;
    
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msgDiv;
}

function showTypingIndicator() {
    const html = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
    return appendMessage('ai', html);
}

// Auto-resize textarea
chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    if (this.value === '') this.style.height = '50px'; // Reset to min-height
});

function handleEnter(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

// --- Text Chat Routing ---
async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    
    // Replace newlines with <br> for display
    const formattedText = text.replace(/\n/g, '<br>');
    appendMessage('user', formattedText);
    
    chatInput.value = '';
    chatInput.style.height = '50px'; // Reset height
    
    const typingBubble = showTypingIndicator();
    
    try {
        const response = await fetch('/api/chat/router', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        typingBubble.remove();
        
        if (data.intent === 'error') {
            appendMessage('ai', data.message);
        } 
        else if (data.intent === 'tabular_form') {
            // Generate inline form!
            let formHtml = `<p>${data.message}</p>
                            <div class="inline-form" id="form-${data.disease}">
                                <div class="form-grid">`;
            
            data.features.forEach(f => {
                formHtml += `<input type="number" step="any" id="feat-${f}" placeholder="${f.replace(/_/g, ' ')}" required>`;
            });
            
            formHtml += `</div>
                         <button onclick="submitTabularForm('${data.disease}')">Analyze Risk</button>
                         </div>`;
            
            appendMessage('ai', formHtml);
            // Save features globally for the submit function
            window.currentDisease = data.disease;
            window.currentFeatures = data.features;
        }
        else if (data.intent === 'symptom_prediction') {
            let reply = `<p>Based on your symptoms (<strong>${data.matched.join(', ')}</strong>), here are my top predictions:</p><ul>`;
            
            data.predictions.forEach(p => {
                reply += `<li><strong>${p.disease}</strong> (${p.confidence}% confidence)
                          <div class="conf-bar-bg"><div class="conf-bar-fill" style="width: 0%"></div></div></li>`;
            });
            reply += `</ul>`;
            
            const msgNode = appendMessage('ai', reply);
            
            // Animate bars
            setTimeout(() => {
                const bars = msgNode.querySelectorAll('.conf-bar-fill');
                data.predictions.forEach((p, i) => {
                    if(bars[i]) bars[i].style.width = p.confidence + '%';
                });
            }, 50);
        }
        
    } catch (err) {
        typingBubble.remove();
        appendMessage('ai', "Sorry, I encountered a network error. " + err.message);
    }
}

// --- Inline Form Submission ---
async function submitTabularForm(diseaseName) {
    const featuresData = {};
    let missing = false;
    
    window.currentFeatures.forEach(f => {
        const val = document.getElementById(`feat-${f}`).value;
        if (val === "") missing = true;
        featuresData[f] = parseFloat(val);
    });
    
    if (missing) {
        alert("Please fill out all fields in the form.");
        return;
    }
    
    // Hide form to show processing
    document.getElementById(`form-${diseaseName}`).innerHTML = `<em>Processing clinical data...</em>`;
    
    const typingBubble = showTypingIndicator();
    
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
        typingBubble.remove();
        
        if (data.error) throw new Error(data.error);
        
        const isSick = data.prediction === 1;
        const color = isSick ? 'var(--danger)' : 'var(--success)';
        
        let reply = `<h4>Clinical Results: ${diseaseName}</h4>
                     <p style="color: ${color}; font-weight: 800; font-size: 1.2rem; margin: 10px 0;">${data.prediction_text}</p>
                     <p>Confidence: ${data.confidence}%</p>
                     <div class="conf-bar-bg"><div class="conf-bar-fill" style="width: 0%; background: ${color}"></div></div>`;
                     
        const msgNode = appendMessage('ai', reply);
        
        setTimeout(() => {
            msgNode.querySelector('.conf-bar-fill').style.width = data.confidence + '%';
        }, 50);
        
    } catch (err) {
        typingBubble.remove();
        appendMessage('ai', "Error processing clinical data: " + err.message);
    }
}

// --- MRI Image Upload ---
let mriFile = null;

function handleImageUpload(event) {
    mriFile = event.target.files[0];
    if (!mriFile) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('mri-preview').src = e.target.result;
        document.getElementById('image-preview-overlay').classList.remove('hidden');
    }
    reader.readAsDataURL(mriFile);
}

function cancelUpload() {
    mriFile = null;
    document.getElementById('mri-upload').value = '';
    document.getElementById('image-preview-overlay').classList.add('hidden');
}

async function confirmUpload() {
    document.getElementById('image-preview-overlay').classList.add('hidden');
    
    // Show image in chat as user message
    const url = URL.createObjectURL(mriFile);
    appendMessage('user', `<img src="${url}" style="max-width: 200px; border-radius: 8px;">`);
    
    const typingBubble = showTypingIndicator();
    
    const formData = new FormData();
    formData.append('image', mriFile);
    
    try {
        const response = await fetch('/api/predict/mri', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        typingBubble.remove();
        
        if (data.error) throw new Error(data.error);
        
        const isHealthy = data.tumor_type.toLowerCase() === 'notumor';
        const color = isHealthy ? 'var(--success)' : 'var(--danger)';
        
        let reply = `<h4>MRI Scan Analysis Complete</h4>
                     <p>Detection: <strong style="color: ${color}">${data.tumor_type.toUpperCase()}</strong></p>
                     <p>Confidence: ${data.confidence}%</p>
                     <div class="conf-bar-bg"><div class="conf-bar-fill" style="width: 0%; background: ${color}"></div></div>`;
                     
        const msgNode = appendMessage('ai', reply);
        
        setTimeout(() => {
            msgNode.querySelector('.conf-bar-fill').style.width = data.confidence + '%';
        }, 50);
        
    } catch (err) {
        typingBubble.remove();
        appendMessage('ai', "Error scanning MRI: " + err.message);
    }
    
    // reset
    mriFile = null;
    document.getElementById('mri-upload').value = '';
}
