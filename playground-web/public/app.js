let availableModels = [];
let isGenerating = false;
let currentTokenizerType = 'hf'; // Default

// DOM Elements
const chatArea = document.getElementById('chatArea');
const messagesContainer = document.getElementById('messagesContainer');
const welcomeView = document.getElementById('welcomeView');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const modelSelector = document.getElementById('modelSelector');
const currentModelName = document.getElementById('currentModelName');
const modelModal = document.getElementById('modelModal');
const modelList = document.getElementById('modelList');
const statusIndicator = document.getElementById('statusIndicator');

// Settings Elements
const settingsBtn = document.getElementById('settingsBtn');
const settingsPanel = document.getElementById('settingsPanel');
const paramMaxTokens = document.getElementById('paramMaxTokens');
const paramTemperature = document.getElementById('paramTemperature');
const valTemperature = document.getElementById('valTemperature');
const paramTopK = document.getElementById('paramTopK');
const paramDoSample = document.getElementById('paramDoSample');
const paramTokenizer = document.getElementById('paramTokenizer');

// Initialize
async function init() {
    setupEventListeners();
    await fetchModels();
}

async function fetchModels() {
    try {
        const response = await fetch('http://localhost:3001/api/models');
        availableModels = await response.json();
        renderModelList();
    } catch (err) {
        console.error('Failed to fetch models:', err);
        statusIndicator.innerText = 'Offline';
    }
}

function renderModelList() {
    modelList.innerHTML = availableModels.map(m => `
        <div class="suggested-card" onclick="selectModelById('${m.id}')">
            <p class="suggested-title">${m.name}</p>
            <p class="suggested-desc">${new Date(m.date).toLocaleString()}</p>
        </div>
    `).join('');
}

window.selectModelById = async (id) => {
    const modelInfo = availableModels.find(m => m.id === id);
    if (modelInfo) await selectModel(modelInfo);
};

async function selectModel(info) {
    closeModal();
    statusIndicator.innerHTML = '<div class="loader"></div> Loading on Server...';
    currentModelName.innerText = info.name;

    currentTokenizerType = paramTokenizer.value;

    try {
        const response = await fetch('http://localhost:3001/api/select-model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: info.id, tokenizerType: currentTokenizerType })
        });

        const result = await response.json();

        statusIndicator.innerText = 'Ready';
        welcomeView.style.display = 'none';
        addMessage('system', `Server successfully loaded model: ${info.name}. Parameters: ${JSON.stringify(result.params)}`);
    } catch (err) {
        console.error(err);
        statusIndicator.innerText = 'Error';
        addMessage('system', 'Error: Failed to load model on server.');
    }
}

function setupEventListeners() {
    modelSelector.onclick = () => {
        modelModal.style.display = 'flex';
        fetchModels(); // Refresh list
    };
    sendBtn.onclick = handleSend;
    userInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
    userInput.onkeydown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    settingsBtn.onclick = toggleSettings;
    paramTemperature.oninput = () => {
        valTemperature.innerText = paramTemperature.value;
    };
    paramTokenizer.onchange = () => {
        currentTokenizerType = paramTokenizer.value;
        // If a model is already loaded, we might want to re-load it to update tokenizer
        // For now, let's just log it
        console.log(`Tokenizer type changed to: ${currentTokenizerType}`);
    };
}

window.toggleSettings = () => {
    const isVisible = settingsPanel.style.display === 'flex';
    settingsPanel.style.display = isVisible ? 'none' : 'flex';
};

window.closeModal = () => modelModal.style.display = 'none';
window.setPrompt = (text) => {
    userInput.value = text;
    userInput.dispatchEvent(new Event('input'));
};

function addMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;
    msgDiv.innerHTML = `
        <div class="avatar" style="background: ${role === 'user' ? '#10b981' : (role === 'system' ? '#444' : '#3b82f6')}"></div>
        <div class="message-content">
            <div style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.25rem; opacity: 0.6;">${role}</div>
            <div class="text-body">${text.replace(/\n/g, '<br>')}</div>
            <div class="stats-area"></div>
        </div>
    `;
    messagesContainer.appendChild(msgDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
    return {
        body: msgDiv.querySelector('.text-body'),
        stats: msgDiv.querySelector('.stats-area')
    };
}

async function handleSend() {
    const text = userInput.value.trim();
    if (!text || isGenerating) return;

    userInput.value = '';
    userInput.dispatchEvent(new Event('input'));
    addMessage('user', text);

    await runStreamingChat(text);
}

async function runStreamingChat(prompt) {
    isGenerating = true;
    sendBtn.classList.add('disabled');
    const { body: responseBody, stats: statsArea } = addMessage('assistant', '');

    let fullText = "";

    // Get current params from UI
    const params = {
        maxTokens: paramMaxTokens.value,
        temperature: paramTemperature.value,
        doSample: paramDoSample.checked,
        topK: paramTopK.value
    };

    const queryParams = new URLSearchParams({
        prompt: prompt,
        ...params
    });

    try {
        const eventSource = new EventSource(`http://localhost:3001/api/chat?${queryParams.toString()}`);

        eventSource.onmessage = (event) => {
            if (event.data === '[DONE]') {
                eventSource.close();
                isGenerating = false;
                sendBtn.classList.remove('disabled');
                return;
            }

            try {
                const data = JSON.parse(event.data);
                if (data.token) {
                    fullText += data.token;
                    responseBody.innerHTML = fullText.replace(/\n/g, '<br>');
                    chatArea.scrollTop = chatArea.scrollHeight;
                }
                if (data.stats) {
                    renderStats(statsArea, data.stats);
                }
                if (data.error) {
                    responseBody.innerHTML += `<br><span style="color:red">[Error: ${data.error}]</span>`;
                    eventSource.close();
                }
            } catch (e) {
                console.error('Parse error:', e);
            }
        };

        eventSource.onerror = (err) => {
            console.error('SSE Error:', err);
            eventSource.close();
            isGenerating = false;
            sendBtn.classList.remove('disabled');
        };

    } catch (err) {
        console.error(err);
        responseBody.innerText = "Error connecting to server.";
        isGenerating = false;
        sendBtn.classList.remove('disabled');
    }
}

function renderStats(container, stats) {
    container.innerHTML = `
        <div class="stats-container">
            <div class="stat-item"><span class="stat-icon">⏱️</span> Generation: ${stats.generationTime}ms</div>
            <div class="stat-item"><span class="stat-icon">📏</span> Tokens: ${stats.tokenCount}</div>
            <div class="stat-item"><span class="stat-icon">🎯</span> ${stats.tokensPerSec} tok/s</div>
            <div class="stat-item"><span class="stat-icon">💾</span> Tensors: ${stats.memory}</div>
        </div>
    `;
    chatArea.scrollTop = chatArea.scrollHeight;
}

init();
