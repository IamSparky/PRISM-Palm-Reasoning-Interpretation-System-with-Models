/* ============================================================
   PRISM PalmRAG — script.js
   Fixes: follow-up glitch, chat history, tab switching
   ============================================================ */

// ── DOM refs ──────────────────────────────────────────────────
const fileInput     = document.getElementById('palm-image');
const previewArea   = document.getElementById('preview-area');
const previewImage  = document.getElementById('image-preview');
const previewFallback = document.getElementById('preview-fallback');
const fileLabelText = document.getElementById('file-label-text');
const sendBtn       = document.getElementById('send-btn');
const chatContainer = document.getElementById('chat-container');
const currentImgBox = document.getElementById('current-image-box');

let currentPreviewUrl = null;  // Object URL for the image currently shown

// ── Session state ─────────────────────────────────────────────
// sessions: Array<{ id, imageDataUrl, imageFileName, messages: [{role, text}], timestamp }>
let sessions      = loadSessions();
let activeSession = null;  // the session being built in the current chat

// ── Boot ──────────────────────────────────────────────────────
renderHistoryList();

// ── File input ────────────────────────────────────────────────
fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (!file) return;

    // Update filename label
    fileLabelText.textContent = truncateFileName(file.name, 14);

    // Show preview in main chat area
    if (currentPreviewUrl) URL.revokeObjectURL(currentPreviewUrl);
    currentPreviewUrl = URL.createObjectURL(file);
    previewImage.onerror = () => {
        previewImage.style.display = 'none';
        previewFallback.hidden = false;
    };
    previewImage.style.display = 'block';
    previewImage.src = currentPreviewUrl;
    previewFallback.hidden = true;
    previewArea.hidden = false;

    // Update left-sidebar thumbnail
    const reader = new FileReader();
    reader.onload = (e) => {
        showSidebarThumb(e.target.result);
        // Start a new session for this new image
        startNewSession(e.target.result, file.name);
    };
    reader.readAsDataURL(file);
});

// ── Enter key ────────────────────────────────────────────────
document.getElementById('user-question').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendRequest();
    }
});

// ── Send request ─────────────────────────────────────────────
async function sendRequest() {
    const questionInput = document.getElementById('user-question');
    const question = questionInput.value.trim();
    if (!question) return;

    // Immediately clear the input field — BEFORE anything async.
    // This prevents the glitch where text "jumps" or re-appears.
    questionInput.value = '';

    // Show user bubble
    appendMessage('user-msg', question);

    // Build FormData
    const formData = new FormData();
    if (fileInput.files[0]) {
        formData.append('image', fileInput.files[0]);
    }
    formData.append('question', question);

    // Lock UI
    sendBtn.disabled = true;
    sendBtn.textContent = '✦ Reading...';
    const loadingId = appendLoadingIndicator();

    try {
        const response = await fetch('http://localhost:8000/analyze', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorBody = await response.json().catch(() => null);
            const errorMsg = errorBody?.detail || `Server error ${response.status}`;
            throw new Error(errorMsg);
        }

        const data = await response.json();
        removeElement(loadingId);
        const answer = data.final_answer || 'No answer was returned.';
        appendMessage('ai-msg', answer, true);

        // Save to active session
        if (activeSession) {
            activeSession.messages.push({ role: 'user', text: question });
            activeSession.messages.push({ role: 'ai', text: answer });
            persistSessions();
            renderHistoryList();
        }

    } catch (err) {
        removeElement(loadingId);
        appendMessage('ai-msg', `**Error:** ${err.message}`, true);
    } finally {
        sendBtn.disabled = false;
        sendBtn.textContent = 'Consult AI';
    }
}

// ── Chat helpers ──────────────────────────────────────────────
function appendMessage(className, text, isMarkdown = false) {
    const id  = 'msg-' + Date.now() + Math.random().toString(36).slice(2);
    const div = document.createElement('div');
    div.className = `message ${className}`;
    div.id = id;
    div.innerHTML = isMarkdown ? renderMarkdown(text) : escapeHTML(text);
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return id;
}

function appendLoadingIndicator() {
    const id  = 'loading-' + Date.now();
    const div = document.createElement('div');
    div.className = 'message ai-msg loading-msg';
    div.id = id;
    div.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return id;
}

function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ── Session Management ────────────────────────────────────────
function startNewSession(imageDataUrl, imageFileName) {
    activeSession = {
        id: 'session-' + Date.now(),
        imageDataUrl,
        imageFileName,
        messages: [],
        timestamp: Date.now(),
    };
    sessions.unshift(activeSession);  // newest first
    persistSessions();
    renderHistoryList();

    // Clear chat for new session
    chatContainer.innerHTML = '';
    appendMessage('ai-msg', `New palm image loaded: **${imageFileName}**. Ask your first question.`, true);
}

function loadSessions() {
    try {
        const raw = localStorage.getItem('prism_sessions');
        return raw ? JSON.parse(raw) : [];
    } catch { return []; }
}

function persistSessions() {
    // Keep max 20 sessions; trim image data URLs to save space
    const toSave = sessions.slice(0, 20);
    try {
        localStorage.setItem('prism_sessions', JSON.stringify(toSave));
    } catch (e) {
        // Storage quota: strip image data from oldest sessions
        const stripped = toSave.map((s, i) => i > 5 ? { ...s, imageDataUrl: null } : s);
        try { localStorage.setItem('prism_sessions', JSON.stringify(stripped)); } catch {}
    }
}

function deleteSession(id, event) {
    event.stopPropagation();  // don't trigger card click
    sessions = sessions.filter(s => s.id !== id);
    if (activeSession && activeSession.id === id) activeSession = null;
    persistSessions();
    renderHistoryList();
}

function clearAllHistory() {
    if (!confirm('Delete all saved session history?')) return;
    sessions = [];
    activeSession = null;
    persistSessions();
    renderHistoryList();
}

function loadSessionIntoChat(session) {
    activeSession = session;
    chatContainer.innerHTML = '';

    // Show the session's image in sidebar
    if (session.imageDataUrl) showSidebarThumb(session.imageDataUrl);

    // Replay all messages
    if (session.messages.length === 0) {
        appendMessage('ai-msg', `Session: **${session.imageFileName}**. Ask a question.`, true);
    } else {
        for (const msg of session.messages) {
            appendMessage(msg.role === 'user' ? 'user-msg' : 'ai-msg', msg.text, msg.role === 'ai');
        }
    }

    // Switch to current tab to see the chat
    switchTab('current');
}

// ── History rendering ─────────────────────────────────────────
function renderHistoryList() {
    const list = document.getElementById('history-list');
    if (!list) return;

    if (sessions.length === 0) {
        list.innerHTML = '<p class="empty-history-msg">No saved sessions yet.</p>';
        return;
    }

    list.innerHTML = '';
    sessions.forEach(session => {
        const card = document.createElement('div');
        card.className = 'session-card';
        card.onclick = () => loadSessionIntoChat(session);

        const lastUserMsg = [...session.messages].reverse().find(m => m.role === 'user');
        const preview = lastUserMsg ? lastUserMsg.text : 'No questions yet';
        const dateStr = new Date(session.timestamp).toLocaleDateString('en-IN', {
            day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit'
        });

        const thumbHtml = session.imageDataUrl
            ? `<img src="${session.imageDataUrl}" alt="palm thumb">`
            : `<div class="session-thumb-placeholder">🖐</div>`;

        card.innerHTML = `
            <div class="session-card-header">
                ${thumbHtml}
                <div class="session-meta">
                    <div class="session-title">${escapeHTML(session.imageFileName || 'Palm session')}</div>
                    <div class="session-date">${dateStr}</div>
                </div>
                <button class="session-delete-btn" title="Delete session" onclick="deleteSession('${session.id}', event)">✕</button>
            </div>
            <div class="session-preview">${escapeHTML(preview)}</div>
        `;
        list.appendChild(card);
    });
}

// ── Sidebar thumbnail ─────────────────────────────────────────
function showSidebarThumb(src) {
    currentImgBox.innerHTML = `<img src="${src}" alt="Current palm">`;
}

// ── Tab switching ─────────────────────────────────────────────
function switchTab(tab) {
    document.getElementById('tab-current').classList.toggle('active', tab === 'current');
    document.getElementById('tab-history').classList.toggle('active', tab === 'history');
    document.getElementById('panel-current').classList.toggle('active', tab === 'current');
    document.getElementById('panel-history').classList.toggle('active', tab === 'history');
}

// ── Utility ───────────────────────────────────────────────────
function truncateFileName(name, max) {
    if (name.length <= max) return name;
    const ext = name.slice(name.lastIndexOf('.'));
    return name.slice(0, max - ext.length - 1) + '…' + ext;
}

function escapeHTML(text) {
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function formatInline(text) {
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/__(.+?)__/g,     '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g,     '<em>$1</em>')
        .replace(/_(.+?)_/g,       '<em>$1</em>')
        .replace(/`([^`]+)`/g,     '<code>$1</code>')
        .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}

function renderMarkdown(markdown) {
    const lines = markdown.split('\n');
    let html = '';
    let inList = false, inOrderedList = false, inBlockquote = false, inCodeBlock = false;

    for (const rawLine of lines) {
        const line = rawLine.trimEnd();

        // Code block toggle
        if (/^```/.test(line)) {
            if (inCodeBlock) { html += '</code></pre>'; inCodeBlock = false; }
            else             { closeLists(); html += '<pre><code>'; inCodeBlock = true; }
            continue;
        }
        if (inCodeBlock) { html += escapeHTML(line) + '\n'; continue; }

        // HR
        if (/^---+$/.test(line.trim())) {
            closeLists(); closeBlockquote();
            html += '<hr>';
            continue;
        }

        // Headings
        const hm = line.match(/^(#{1,6})\s+(.*)$/);
        if (hm) {
            closeLists(); closeBlockquote();
            html += `<h${hm[1].length}>${formatInline(hm[2])}</h${hm[1].length}>`;
            continue;
        }

        // Blockquote
        const bm = line.match(/^>\s?(.*)$/);
        if (bm) {
            if (!inBlockquote) { closeLists(); html += '<blockquote>'; inBlockquote = true; }
            html += `<p>${formatInline(bm[1])}</p>`;
            continue;
        }

        // Ordered list
        const om = line.match(/^\d+\.\s+(.*)$/);
        if (om) {
            if (!inOrderedList) { closeLists(); html += '<ol>'; inOrderedList = true; }
            html += `<li>${formatInline(om[1])}</li>`;
            continue;
        }

        // Unordered list
        const um = line.match(/^[-*+]\s+(.*)$/);
        if (um) {
            if (!inList) { closeLists(); html += '<ul>'; inList = true; }
            html += `<li>${formatInline(um[1])}</li>`;
            continue;
        }

        // Empty line
        if (line === '') {
            closeLists(); closeBlockquote();
            continue;
        }

        // Paragraph
        closeLists(); closeBlockquote();
        html += `<p>${formatInline(line)}</p>`;
    }

    closeLists(); closeBlockquote();
    if (inCodeBlock) html += '</code></pre>';
    return html;

    function closeLists() {
        if (inList)        { html += '</ul>'; inList = false; }
        if (inOrderedList) { html += '</ol>'; inOrderedList = false; }
    }
    function closeBlockquote() {
        if (inBlockquote) { html += '</blockquote>'; inBlockquote = false; }
    }
}
