// ==========================================================================
// Football RAG HUD Dashboard JavaScript Client
// ==========================================================================

const API_BASE = '/api/v1';
const GRAPHQL_URL = '/graphql';

// State variables
let token = localStorage.getItem('access_token') || null;
let currentTab = 'predictions-tab';
let selectedLeague = 'Premier_League';
let conversations = [];
let activeConversationId = null;

// On startup
document.addEventListener('DOMContentLoaded', () => {
    try { initTabNavigation(); } catch(e) { console.error('initTabNavigation failed:', e); }
    try { checkAuthStatus(); } catch(e) { console.error('checkAuthStatus failed:', e); }
    try { loadLeagueTeams(selectedLeague); } catch(e) { console.error('loadLeagueTeams failed:', e); }
    try { setupEventListeners(); } catch(e) { console.error('setupEventListeners failed:', e); }
    console.log('DOMContentLoaded: all init functions attempted');
});

// Navigation handlers
function initTabNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            
            const targetTab = item.getAttribute('data-tab');
            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
            document.getElementById(targetTab).classList.add('active');
            
            currentTab = targetTab;
            if (currentTab === 'chat-tab') {
                loadConversations();
            }
            if (currentTab === 'contribute-tab') {
                const contributeLeague = document.getElementById('match-league-select').value;
                loadContributeTeams(contributeLeague);
            }
            if (currentTab === 'best11-tab') {
                const best11League = document.getElementById('best11-league-select').value;
                loadBest11Teams(best11League);
            }
            if (currentTab === 'supervisor-tab') {
                fetchSupervisorQueue();
            }
        });
    });
}

// Authentication & Token management
async function checkAuthStatus() {
    const userDisplay = document.getElementById('user-display');
    const authBtn = document.getElementById('auth-btn');
    const supervisorEls = document.querySelectorAll('.supervisor-only');
    
    const hideSupervisorElements = () => {
        supervisorEls.forEach(el => { el.style.display = 'none'; });
    };
    const showSupervisorElements = () => {
        supervisorEls.forEach(el => { el.style.display = ''; });
    };
    
    if (token) {
        try {
            const res = await fetch(`${API_BASE}/auth/me`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.ok) {
                const user = await res.json();
                userDisplay.textContent = `USER: ${user.username.toUpperCase()} (${user.role.toUpperCase()})`;
                userDisplay.classList.add('user-display-link');
                authBtn.textContent = 'LOGOUT';
                authBtn.classList.remove('accent-border');
                authBtn.classList.add('muted-btn');
                updateAuthModalTabs(true);
                if (user.role === 'supervisor') {
                    showSupervisorElements();
                } else {
                    hideSupervisorElements();
                }
                return;
            }
        } catch (e) {
            console.error("Auth check failed", e);
        }
    }
    
    // Default to unauthenticated layout
    localStorage.removeItem('access_token');
    token = null;
    userDisplay.textContent = 'UNAUTHENTICATED';
    userDisplay.classList.remove('user-display-link');
    authBtn.textContent = 'LOGIN / REGISTER';
    authBtn.classList.remove('muted-btn');
    authBtn.classList.add('accent-border');
    updateAuthModalTabs(false);
    hideSupervisorElements();
}

function updateAuthModalTabs(isAuthenticated) {
    const tabLogin = document.getElementById('tab-login');
    const tabRegister = document.getElementById('tab-register');
    const tabChangePassword = document.getElementById('tab-change-password');

    tabLogin.style.display = isAuthenticated ? 'none' : '';
    tabRegister.style.display = isAuthenticated ? 'none' : '';
    tabChangePassword.style.display = isAuthenticated ? '' : 'none';
}

function showAuthModal(defaultTab = null) {
    document.getElementById('auth-modal').style.display = 'flex';
    if (token) {
        switchAuthTab('changepass');
    } else {
        switchAuthTab(defaultTab || 'login');
    }
}

function hideAuthModal() {
    document.getElementById('auth-modal').style.display = 'none';
    document.getElementById('login-form').reset();
    document.getElementById('register-form').reset();
    document.getElementById('change-password-form').reset();
}

function switchAuthTab(tab) {
    const tabLogin = document.getElementById('tab-login');
    const tabRegister = document.getElementById('tab-register');
    const tabChangePassword = document.getElementById('tab-change-password');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    const changePasswordForm = document.getElementById('change-password-form');

    const tabs = { login: tabLogin, register: tabRegister, changepass: tabChangePassword };
    const forms = { login: loginForm, register: registerForm, changepass: changePasswordForm };

    Object.entries(tabs).forEach(([name, el]) => {
        el.classList.toggle('active', name === tab);
    });
    Object.entries(forms).forEach(([name, el]) => {
        const isActive = name === tab;
        el.classList.toggle('active', isActive);
        el.style.display = isActive ? 'block' : 'none';
    });
}

async function handleLogin(e) {
    e.preventDefault();
    const usernameInput = document.getElementById('login-username').value;
    const passwordInput = document.getElementById('login-password').value;
    
    const formData = new URLSearchParams();
    formData.append('username', usernameInput);
    formData.append('password', passwordInput);
    
    try {
        const res = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: formData
        });
        
        if (res.ok) {
            const data = await res.json();
            token = data.access_token;
            localStorage.setItem('access_token', token);
            await checkAuthStatus();
            hideAuthModal();
            // Reload context
            loadLeagueTeams(selectedLeague);
        } else {
            const err = await res.json();
            alert(`Authentication failed: ${err.detail || 'Invalid credentials'}`);
        }
    } catch (e) {
        console.error(e);
        alert('Error contacting authentication server.');
    }
}

async function handleRegister(e) {
    e.preventDefault();
    const username = document.getElementById('reg-username').value;
    const email = document.getElementById('reg-email').value;
    const password = document.getElementById('reg-password').value;
    const role = document.getElementById('reg-role').value;
    
    try {
        const res = await fetch(`${API_BASE}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password, role })
        });
        
        if (res.ok) {
            alert('Registration successful! Please log in.');
            switchAuthTab('login');
            // Auto-fill login username
            document.getElementById('login-username').value = username;
        } else {
            const err = await res.json();
            alert(`Registration failed: ${err.detail || 'Unknown error'}`);
        }
    } catch (e) {
        console.error(e);
        alert('Error contacting server during registration.');
    }
}

async function handleChangePassword(e) {
    e.preventDefault();
    const currentPassword = document.getElementById('current-password').value;
    const newPassword = document.getElementById('new-password').value;
    const confirmPassword = document.getElementById('confirm-password').value;

    if (newPassword !== confirmPassword) {
        alert('New password and confirmation do not match.');
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/auth/change-password`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                current_password: currentPassword,
                new_password: newPassword
            })
        });

        if (res.ok) {
            alert('Password updated successfully.');
            hideAuthModal();
        } else {
            const err = await res.json();
            alert(`Password change failed: ${err.detail || 'Unknown error'}`);
        }
    } catch (e) {
        console.error(e);
        alert('Error contacting authentication server.');
    }
}

// GraphQL team fetching
async function graphqlQuery(query, variables = {}) {
    const headers = { 'Content-Type': 'application/json' };
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    const res = await fetch(GRAPHQL_URL, {
        method: 'POST',
        headers,
        body: JSON.stringify({ query, variables })
    });
    const result = await res.json();
    return result.data;
}

async function loadLeagueTeams(league) {
    const query = `
        query GetTeams($league: String!) {
            leagueTeams(league: $league)
        }
    `;
    try {
        const data = await graphqlQuery(query, { league });
        const teams = data ? data.leagueTeams : [];
        
        // Populate selectors
        populateTeamSelects(teams);
    } catch (e) {
        console.error("Failed to load league teams via GraphQL", e);
    }
}

function populateTeamSelects(teams) {
    const homeSelect = document.getElementById('home-team-select');
    const awaySelect = document.getElementById('away-team-select');
    const profileSelect = document.getElementById('profile-team-select');
    
    const selects = [homeSelect, awaySelect, profileSelect];
    selects.forEach(select => {
        if (!select) return;
        select.innerHTML = '';
        teams.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = t.replace(/_/g, ' ');
            select.appendChild(opt);
        });
    });
    
    // Set default distinct selections if possible
    if (teams.length >= 2) {
        if (homeSelect) homeSelect.selectedIndex = 0;
        if (awaySelect) awaySelect.selectedIndex = 1;
    }
}

// setup Event Listeners
function setupEventListeners() {
    // Auth button click (login/register overlay or logout)
    document.getElementById('auth-btn').addEventListener('click', () => {
        if (token) {
            // Logout action
            localStorage.removeItem('access_token');
            token = null;
            checkAuthStatus();
            // Reload context
            loadLeagueTeams(selectedLeague);
        } else {
            showAuthModal();
        }
    });

    // Open account modal (change password tab) when logged in
    document.getElementById('user-display').addEventListener('click', () => {
        if (token) {
            showAuthModal('changepass');
        }
    });

    // Close auth modal
    document.getElementById('close-modal-btn').addEventListener('click', hideAuthModal);
    
    // Close modal on background click
    document.getElementById('auth-modal').addEventListener('click', (e) => {
        if (e.target === document.getElementById('auth-modal')) {
            hideAuthModal();
        }
    });

    // Switch tabs
    document.getElementById('tab-login').addEventListener('click', () => switchAuthTab('login'));
    document.getElementById('tab-register').addEventListener('click', () => switchAuthTab('register'));
    document.getElementById('tab-change-password').addEventListener('click', () => switchAuthTab('changepass'));

    // Forms
    document.getElementById('login-form').addEventListener('submit', handleLogin);
    document.getElementById('register-form').addEventListener('submit', handleRegister);
    document.getElementById('change-password-form').addEventListener('submit', handleChangePassword);
    
    // League picker in sandbox
    document.getElementById('league-select').addEventListener('change', (e) => {
        selectedLeague = e.target.value;
        loadLeagueTeams(selectedLeague);
    });
    
    // League picker in profile
    document.getElementById('profile-league-select').addEventListener('change', async (e) => {
        const league = e.target.value;
        const query = `
            query GetTeams($league: String!) {
                leagueTeams(league: $league)
            }
        `;
        const data = await graphqlQuery(query, { league });
        const teams = data ? data.leagueTeams : [];
        const profileSelect = document.getElementById('profile-team-select');
        profileSelect.innerHTML = '';
        teams.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = t.replace(/_/g, ' ');
            profileSelect.appendChild(opt);
        });
    });
    
    // Run prediction button
    document.getElementById('run-prediction-btn').addEventListener('click', runMatchPrediction);
    
    // Chat stream buttons
    document.getElementById('new-chat-btn').addEventListener('click', createNewConversation);
    document.getElementById('chat-send-btn').addEventListener('click', sendChatMessage);
    document.getElementById('chat-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
    
    // Retrieve profile button
    document.getElementById('load-profile-btn').addEventListener('click', loadTeamProfile);

    // Best XI tab
    document.getElementById('best11-league-select').addEventListener('change', (e) => {
        loadBest11Teams(e.target.value);
    });
    document.getElementById('predict-best11-btn').addEventListener('click', predictBest11);
    
    // Contribute tab: league change → reload team selects
    document.getElementById('match-league-select').addEventListener('change', async (e) => {
        loadContributeTeams(e.target.value);
    });
    
    // Submit match record
    document.getElementById('submit-match-btn').addEventListener('click', submitMatchRecord);
    
    // Submit tactical analysis
    document.getElementById('submit-tactical-btn').addEventListener('click', submitTacticalAnalysis);
    
    // Submit team profile edit
    document.getElementById('submit-profile-btn').addEventListener('click', submitTeamProfileEdit);
}

// ─────────────────────────────────────────────────────────────────────────────
// Business Logic Functions
// ─────────────────────────────────────────────────────────────────────────────

// Match Predictions
async function runMatchPrediction() {
    const home = document.getElementById('home-team-select').value;
    const away = document.getElementById('away-team-select').value;
    const matchDate = document.getElementById('match-date').value;
    const container = document.getElementById('prediction-output-container');
    
    if (!token) {
        alert("You must login first using the QUICK LOGIN button to run predictions.");
        return;
    }
    
    container.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">⏳</div>
            <div>COMPUTING HYBRID PREDICTIONS...</div>
            <div class="empty-desc">Loading GNN edge features and executing LLM contextual analysis.</div>
        </div>
    `;
    
    try {
        const res = await fetch(`${API_BASE}/predictions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                home_team: home,
                away_team: away,
                match_date: matchDate
            })
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Prediction request failed");
        }
        
        const data = await res.json();
        
        // Formulate probability predictions: Mock outcome probabilities since actual backend might have different outputs
        // Typically GNN outputs a prediction result like H, D, or A.
        const rawRes = data.predicted_result;
        const verdict = (rawRes === 'H' || rawRes === 'Home Win') ? 'HOME WIN' : ((rawRes === 'A' || rawRes === 'Away Win') ? 'AWAY WIN' : 'DRAW');
        
        // Dynamically compute confidence rating based on highest probability
        let maxProb = 0;
        if (data.probabilities) {
            maxProb = Math.max(data.probabilities.H || 0, data.probabilities.D || 0, data.probabilities.A || 0);
        }
        let confidenceRating = "Medium";
        if (maxProb >= 0.60) {
            confidenceRating = "High";
        } else if (maxProb < 0.45) {
            confidenceRating = "Low";
        }

        // Parse the LLM response if it is structured JSON
        let analysis = {
            prediction_verdict: data.tactical_analysis,
            confidence_rating: confidenceRating,
            home_team_analysis: { strengths: ["Form continuity", "Expected tactical alignment"], weaknesses: ["Transition vulnerability"] },
            away_team_analysis: { strengths: ["High press efficiency", "Set piece delivery"], weaknesses: ["Defensive depth control"] },
            tactical_matchup_summary: data.tactical_analysis
        };
        
        try {
            // Check if backend returned JSON string inside tactical_analysis
            const parsed = JSON.parse(data.tactical_analysis);
            if (parsed && typeof parsed === 'object') {
                analysis = { ...analysis, ...parsed };
            }
        } catch (e) {
            // Keep original plain text if it failed to parse
        }
        
        const probs = data.probabilities || { H: 0.33, D: 0.34, A: 0.33 };
        const homePct = Math.round(probs.H * 100) + '%';
        const drawPct = Math.round(probs.D * 100) + '%';
        const awayPct = Math.round(probs.A * 100) + '%';
        
        // Render Result UI
        container.innerHTML = `
            <div class="pred-header-card">
                <div class="pred-title">
                    <span class="pred-vs">${home.replace(/_/g, ' ')} <span class="vs-divider">VS</span> ${away.replace(/_/g, ' ')}</span>
                    <span class="verdict-tag">${verdict}</span>
                </div>
                <div class="probability-container">
                    <div class="prob-row">
                        <span class="prob-label">HOME</span>
                        <div class="prob-bar-wrapper">
                            <div class="prob-bar-fill" style="width: ${homePct}"></div>
                        </div>
                        <span class="prob-val">${homePct}</span>
                    </div>
                    <div class="prob-row">
                        <span class="prob-label">DRAW</span>
                        <div class="prob-bar-wrapper">
                            <div class="prob-bar-fill" style="width: ${drawPct}"></div>
                        </div>
                        <span class="prob-val">${drawPct}</span>
                    </div>
                    <div class="prob-row">
                        <span class="prob-label">AWAY</span>
                        <div class="prob-bar-wrapper">
                            <div class="prob-bar-fill" style="width: ${awayPct}"></div>
                        </div>
                        <span class="prob-val">${awayPct}</span>
                    </div>
                </div>
            </div>

            <div class="analysis-grid">
                <div class="analysis-card full-width">
                    <div class="card-title">EXPERT THESIS</div>
                    <div class="card-body">
                        <p style="font-weight: 600; margin-bottom: 8px;">Confidence: <span class="accent-text">${analysis.confidence_rating}</span></p>
                        <p>${analysis.prediction_verdict}</p>
                    </div>
                </div>
                
                <div class="analysis-card">
                    <div class="card-title">${home.replace(/_/g, ' ').toUpperCase()} SCOUTING</div>
                    <div class="card-body">
                        <p style="color: var(--accent); margin-bottom: 6px; font-weight: bold; font-size: 0.75rem;">STRENGTHS</p>
                        <ul class="card-list" style="margin-bottom: 12px;">
                            ${analysis.home_team_analysis.strengths.map(s => `<li>${s}</li>`).join('')}
                        </ul>
                        <p style="color: #ff6600; margin-bottom: 6px; font-weight: bold; font-size: 0.75rem;">WEAKNESSES</p>
                        <ul class="card-list">
                            ${analysis.home_team_analysis.weaknesses.map(w => `<li>${w}</li>`).join('')}
                        </ul>
                    </div>
                </div>
                
                <div class="analysis-card">
                    <div class="card-title">${away.replace(/_/g, ' ').toUpperCase()} SCOUTING</div>
                    <div class="card-body">
                        <p style="color: var(--accent); margin-bottom: 6px; font-weight: bold; font-size: 0.75rem;">STRENGTHS</p>
                        <ul class="card-list" style="margin-bottom: 12px;">
                            ${analysis.away_team_analysis.strengths.map(s => `<li>${s}</li>`).join('')}
                        </ul>
                        <p style="color: #ff6600; margin-bottom: 6px; font-weight: bold; font-size: 0.75rem;">WEAKNESSES</p>
                        <ul class="card-list">
                            ${analysis.away_team_analysis.weaknesses.map(w => `<li>${w}</li>`).join('')}
                        </ul>
                    </div>
                </div>

                <div class="analysis-card full-width">
                    <div class="card-title">TACTICAL ENCOUNTER SUMMARY</div>
                    <div class="card-body">
                        <p>${analysis.tactical_matchup_summary || 'Analysis details not parsed.'}</p>
                    </div>
                </div>
            </div>
        `;
        
    } catch (e) {
        container.innerHTML = `
            <div class="empty-state" style="color: #ff3333">
                <div class="empty-icon">❌</div>
                <div>ERROR COMPUTING PREDICTION</div>
                <div class="empty-desc">${e.message}</div>
            </div>
        `;
    }
}

// Conversations & Chatbot
async function loadConversations() {
    const listContainer = document.getElementById('conv-list-container');
    if (!token) {
        listContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 0.65rem; text-align:center;">LOGIN REQUIRED</div>';
        return;
    }
    
    try {
        const res = await fetch(`${API_BASE}/chat/conversations`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        if (res.ok) {
            conversations = await res.json();
            renderConversationsList();
        }
    } catch (e) {
        console.error(e);
    }
}

function renderConversationsList() {
    const listContainer = document.getElementById('conv-list-container');
    listContainer.innerHTML = '';
    
    conversations.forEach(c => {
        const btn = document.createElement('button');
        btn.className = `conv-item ${c.id === activeConversationId ? 'active' : ''}`;
        btn.textContent = `[${c.mode.toUpperCase()}] ${c.title}`;
        btn.addEventListener('click', () => selectConversation(c.id));
        listContainer.appendChild(btn);
    });
}

async function selectConversation(id) {
    activeConversationId = id;
    renderConversationsList();
    
    const messagesContainer = document.getElementById('chat-messages-container');
    messagesContainer.innerHTML = '<div class="empty-state"><div class="empty-icon">⏳</div><div>LOADING THREAD HISTORY...</div></div>';
    
    try {
        const res = await fetch(`${API_BASE}/chat/conversations/${id}/messages`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        if (res.ok) {
            const messages = await res.json();
            renderMessages(messages);
        }
    } catch (e) {
        console.error(e);
    }
}

function renderMessages(messages) {
    const messagesContainer = document.getElementById('chat-messages-container');
    messagesContainer.innerHTML = '';
    
    if (messages.length === 0) {
        messagesContainer.innerHTML = '<div class="empty-state"><div>NO MESSAGES IN THIS THREAD</div><div class="empty-desc">Ask your first question below.</div></div>';
        return;
    }
    
    messages.forEach(m => {
        const div = document.createElement('div');
        div.className = `chat-msg ${m.sender}`;
        div.textContent = m.content;
        if (m.sender === 'assistant' && Array.isArray(m.sources) && m.sources.length > 0) {
            const chipRow = document.createElement('div');
            chipRow.className = 'chat-sources';
            m.sources.slice(0, 6).forEach(s => {
                const chip = document.createElement('span');
                chip.className = 'source-chip';
                chip.title = `${s.source_type || ''}${s.league ? ' | ' + s.league : ''}${s.season ? ' ' + s.season : ''}${s.team ? ' | ' + s.team : ''}`;
                chip.textContent = `[${s.ref || '?'}] ${s.title || s.source_type || 'source'}`;
                chipRow.appendChild(chip);
            });
            div.appendChild(chipRow);
        }
        messagesContainer.appendChild(div);
    });
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

async function createNewConversation() {
    if (!token) {
        alert("Please login first.");
        return;
    }
    
    const title = prompt("Enter thread title:", "Tactical Discussion");
    if (!title) return;
    
    try {
        const res = await fetch(`${API_BASE}/chat/conversations`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                title: title,
                mode: 'general'
            })
        });
        if (res.ok) {
            const newConv = await res.json();
            activeConversationId = newConv.id;
            await loadConversations();
            selectConversation(newConv.id);
        }
    } catch (e) {
        console.error(e);
    }
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');
    const content = input.value.trim();
    if (!content) return;
    
    if (!token) {
        alert("Please login first to use the Tactical Chat.");
        showAuthModal();
        return;
    }
    
    // Disable inputs immediately so user knows it is loading and can't double-submit
    input.disabled = true;
    sendBtn.disabled = true;
    sendBtn.textContent = '...';
    
    // If no active thread, automatically initialize a new conversation first
    if (!activeConversationId) {
        const title = content.length > 25 ? content.substring(0, 25) + "..." : content;
        try {
            const res = await fetch(`${API_BASE}/chat/conversations`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    title: title,
                    mode: 'general'
                })
            });
            if (res.ok) {
                const newConv = await res.json();
                activeConversationId = newConv.id;
                await loadConversations();
                selectConversation(newConv.id);
            } else {
                alert("Failed to initialize a new conversation thread.");
                input.disabled = false;
                sendBtn.disabled = false;
                sendBtn.textContent = 'SEND';
                return;
            }
        } catch (e) {
            console.error(e);
            alert("Error creating conversation thread.");
            input.disabled = false;
            sendBtn.disabled = false;
            sendBtn.textContent = 'SEND';
            return;
        }
    }
    
    // Clear input
    input.value = '';
    
    // Add user message to UI immediately for feedback
    const messagesContainer = document.getElementById('chat-messages-container');
    const userDiv = document.createElement('div');
    userDiv.className = 'chat-msg user';
    userDiv.textContent = content;
    messagesContainer.appendChild(userDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Add temporary pulsing typing indicator
    const loaderDiv = document.createElement('div');
    loaderDiv.className = 'chat-msg assistant typing-indicator';
    loaderDiv.innerHTML = '<span></span><span></span><span></span>';
    messagesContainer.appendChild(loaderDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    try {
        const res = await fetch(`${API_BASE}/chat/conversations/${activeConversationId}/messages`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ content })
        });
        if (res.ok) {
            const data = await res.json();
            // Remove typing indicator class and insert actual response text
            loaderDiv.classList.remove('typing-indicator');
            loaderDiv.textContent = data.content;
        } else {
            loaderDiv.classList.remove('typing-indicator');
            loaderDiv.textContent = "[Error: Failed to fetch response]";
        }
    } catch (e) {
        console.error(e);
        loaderDiv.classList.remove('typing-indicator');
        loaderDiv.textContent = `[Error: ${e.message}]`;
    }
    
    // Re-enable inputs
    input.disabled = false;
    sendBtn.disabled = false;
    sendBtn.textContent = 'SEND';
    input.focus();
    
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Retrieve Team profile via GraphQL
async function loadTeamProfile() {
    const teamName = document.getElementById('profile-team-select').value;
    const container = document.getElementById('team-profile-container');
    
    container.innerHTML = `
        <div class="panel-header">TACTICAL DOSSIER</div>
        <div class="panel-body">
            <div class="empty-state">
                <div class="empty-icon">⏳</div>
                <div>FETCHING DOSSIER VIA GRAPHQL...</div>
            </div>
        </div>
    `;
    
    const query = `
        query GetTeamProfile($name: String!) {
            teamProfile(name: $name) {
                name
                league
                totalMatches
                winRate
                drawRate
                lossRate
                cleanSheetRate
                avgGoalsHome
                avgGoalsAway
                avgXg
                avgXga
                avgShots
                avgShotsAgainst
                avgSot
                avgSotAgainst
                avgCorners
                avgFouls
                avgYellows
                attackTactic
                defenseTactic
                attackHeadline
                defenseHeadline
                strengths
                weaknesses
            }
        }
    `;
    
    try {
        const data = await graphqlQuery(query, { name: teamName });
        const p = data ? data.teamProfile : null;
        
        if (!p) {
            container.innerHTML = `
                <div class="panel-header">TACTICAL DOSSIER</div>
                <div class="panel-body">
                    <div class="empty-state">
                        <div class="empty-icon">⚠️</div>
                        <div>NO GRAPH DATA FOR THIS TEAM</div>
                    </div>
                </div>
            `;
            return;
        }
        
        container.innerHTML = `
            <div class="panel-header">TACTICAL DOSSIER - ${p.name.replace(/_/g, ' ').toUpperCase()}</div>
            <div class="panel-body">
                <div class="radar-stats-grid">
                    <div class="radar-stat-box">
                        <div class="radar-stat-label">WIN RATE</div>
                        <div class="radar-stat-val">${(p.winRate * 100).toFixed(1)}%</div>
                    </div>
                    <div class="radar-stat-box">
                        <div class="radar-stat-label">CLEAN SHEETS</div>
                        <div class="radar-stat-val">${(p.cleanSheetRate * 100).toFixed(1)}%</div>
                    </div>
                    <div class="radar-stat-box">
                        <div class="radar-stat-label">AVG xG</div>
                        <div class="radar-stat-val">${p.avgXg.toFixed(2)}</div>
                    </div>
                    <div class="radar-stat-box">
                        <div class="radar-stat-label">AVG xGA</div>
                        <div class="radar-stat-val">${p.avgXga.toFixed(2)}</div>
                    </div>
                </div>

                <div class="analysis-grid">
                    <div class="analysis-card">
                        <div class="card-title">ATTACK STRUCTURE</div>
                        <div class="card-body">
                            <p style="font-weight: 600; margin-bottom: 8px;" class="accent-text">${p.attackHeadline || 'Tactical build-up'}</p>
                            <p style="margin-bottom: 12px; font-size: 0.8rem; color: var(--text-muted);">${p.attackTactic || 'N/A'}</p>
                            <p style="color: var(--accent); margin-bottom: 6px; font-weight: bold; font-size: 0.75rem;">KEY STRENGTHS</p>
                            <ul class="card-list">
                                ${p.strengths.length ? p.strengths.map(s => `<li>${s}</li>`).join('') : '<li>Possession efficiency</li>'}
                            </ul>
                        </div>
                    </div>

                    <div class="analysis-card">
                        <div class="card-title">DEFENSIVE STRUCTURE</div>
                        <div class="card-body">
                            <p style="font-weight: 600; margin-bottom: 8px; color: #ff6600;">${p.defenseHeadline || 'Defense shape'}</p>
                            <p style="margin-bottom: 12px; font-size: 0.8rem; color: var(--text-muted);">${p.defenseTactic || 'N/A'}</p>
                            <p style="color: #ff6600; margin-bottom: 6px; font-weight: bold; font-size: 0.75rem;">VULNERABILITIES</p>
                            <ul class="card-list">
                                ${p.weaknesses.length ? p.weaknesses.map(w => `<li>${w}</li>`).join('') : '<li>High-line transitions</li>'}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        `;
    } catch (e) {
        console.error(e);
        container.innerHTML = `
            <div class="panel-header">TACTICAL DOSSIER</div>
            <div class="panel-body">
                <div class="empty-state">
                    <div class="empty-icon">❌</div>
                    <div>ERROR RETRIEVING DOSSIER</div>
                    <div class="empty-desc">${e.message}</div>
                </div>
            </div>
        `;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Best XI Tab — Next-Season Preview (predicted XIs + subs, no fixture data)
// ─────────────────────────────────────────────────────────────────────────────

// Data season for predictions = last completed season; the preview targets
// the teams' next-season meeting.
const BEST11_DATA_SEASON = '2425';
const BEST11_PREVIEW_LABEL = '2025-26 SEASON PREVIEW';

async function loadBest11Teams(league) {
    const query = `query GetTeams($league: String!, $season: String) { leagueTeams(league: $league, season: $season) }`;
    const data = await graphqlQuery(query, { league, season: BEST11_DATA_SEASON });
    const teams = data ? data.leagueTeams : [];
    ['best11-home-team-select', 'best11-away-team-select'].forEach(sid => {
        const sel = document.getElementById(sid);
        sel.innerHTML = '';
        teams.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = t.replace(/_/g, ' ');
            sel.appendChild(opt);
        });
    });
    if (teams.length >= 2) {
        document.getElementById('best11-home-team-select').selectedIndex = 0;
        document.getElementById('best11-away-team-select').selectedIndex = 1;
    }
}

async function predictBest11() {
    const league = document.getElementById('best11-league-select').value;
    const home = document.getElementById('best11-home-team-select').value;
    const away = document.getElementById('best11-away-team-select').value;
    const formation = 'auto';
    const container = document.getElementById('best11-result-container');

    if (!home || !away || home === away) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">⚠️</div><div>PICK TWO DIFFERENT TEAMS</div></div>';
        return;
    }
    container.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">⏳</div>
            <div>COMPUTING PREVIEW FOR ${home.replace(/_/g, ' ')} &amp; ${away.replace(/_/g, ' ')}</div>
            <div class="empty-desc">Team-share player ratings · best-fit formation · 2024-25 season games</div>
        </div>
    `;

    const query = `
        query Best11($team: String!, $league: String!, $season: String!, $formation: String!, $opponent: String) {
            best11(team: $team, league: $league, season: $season, formation: $formation, opponent: $opponent) {
                team leagueCode season formation error
                lineup { slot name position rating minutes flex
                         season { goals assists xg xa shots }
                         h2h { matches minutes goals assists xg xa shots } }
                captain
                subs { slot out in ratingDelta reason }
                notes
            }
        }
    `;
    try {
        const baseVars = { team: home, league, season: BEST11_DATA_SEASON, formation, opponent: away };
        const [homeData, awayData] = await Promise.all([
            graphqlQuery(query, baseVars),
            graphqlQuery(query, { ...baseVars, team: away, opponent: home }),
        ]);
        renderBest11Panels(
            homeData ? homeData.best11 : null,
            awayData ? awayData.best11 : null,
            home, away, formation);
    } catch (e) {
        console.error(e);
        container.innerHTML = `<div class="empty-state" style="color:#ff3333"><div class="empty-icon">❌</div><div>ERROR COMPUTING PREVIEW</div><div class="empty-desc">${e.message}</div></div>`;
    }
}

function xiStrength(result) {
    if (!result || !result.lineup || !result.lineup.length) return null;
    const sum = result.lineup.reduce((acc, e) => acc + (e.rating || 0), 0);
    return sum / result.lineup.length;
}

function renderBest11Panels(homeResult, awayResult, homeTeam, awayTeam, formation) {
    const container = document.getElementById('best11-result-container');
    const homeStrength = xiStrength(homeResult);
    const awayStrength = xiStrength(awayResult);
    let verdictLine = '';
    if (homeStrength != null && awayStrength != null) {
        const diff = homeStrength - awayStrength;
        const margin = Math.abs(diff).toFixed(1);
        if (margin < 1.0) {
            verdictLine = `<span class="verdict-chip balanced">BALANCED</span><span class="verdict-margin">XI strength within ${margin} pts</span>`;
        } else {
            const fav = diff > 0 ? homeTeam.replace(/_/g, ' ') : awayTeam.replace(/_/g, ' ');
            verdictLine = `<span class="verdict-chip">${fav.toUpperCase()} FAVORED</span><span class="verdict-margin">XI strength edge +${margin} pts</span>`;
        }
    }
    const header = `
        <div class="pred-header-card">
            <div class="pred-title">
                <span class="pred-vs">${homeTeam.replace(/_/g, ' ')} <span class="vs-divider">VS</span> ${awayTeam.replace(/_/g, ' ')}</span>
                <span class="verdict-tag">${BEST11_PREVIEW_LABEL}</span>
            </div>
            <div class="verdict-row">${verdictLine}</div>
            <div style="font-size:0.65rem; color: var(--text-muted); padding: 0 14px 8px;">
                PREDICTED LINEUPS · AUTO-FIT FORMATIONS · TEAM-SHARE RATINGS (2024-25 SEASON GAMES) · ★ CAPTAIN
            </div>
        </div>
    `;
    container.innerHTML = header + `
        <div class="analysis-grid">
            ${predictionPanel(homeResult, homeTeam, awayTeam)}
            ${predictionPanel(awayResult, awayTeam, homeTeam)}
        </div>
    `;
}

function predictionPanel(result, teamName, opponent) {
    const label = teamName.replace(/_/g, ' ').toUpperCase();
    const oppName = opponent ? opponent.replace(/_/g, ' ') : '';
    if (!result) {
        return `<div class="analysis-card"><div class="card-title">${label}</div><div class="card-body" style="color:#ff3333">NO RESULT</div></div>`;
    }
    if (result.error) {
        return `<div class="analysis-card"><div class="card-title">${label}</div><div class="card-body" style="color:#ff6600">⚠ ${result.error}</div></div>`;
    }
    const order = ['GK', 'DF', 'MF', 'FW'];
    const slots = order
        .map(slot => {
            const entries = result.lineup.filter(e => e.slot === slot);
            if (!entries.length) return '';
            const chips = entries.map(e => {
                const star = e.name === result.captain ? ' <span class="captain-mark" title="captain">★</span>' : '';
                const flexMark = e.flex ? ' <span style="color:#ff6600" title="flex pick">▲</span>' : '';
                const s = e.season || {};
                const sLine = `season ${s.goals ?? 0}G ${s.assists ?? 0}A xG ${(s.xg ?? 0).toFixed(1)}`;
                return `<div class="xi-chip" title="${e.name} — ${e.rating.toFixed(0)}/100 · ${e.minutes} min · ${sLine}">${e.name}${star}${flexMark}<span class="xi-rating">${e.rating.toFixed(0)}</span></div>`;
            }).join('');
            return `<div class="xi-row"><span class="xi-slot">${slot}</span><span class="xi-chips">${chips}</span></div>`;
        })
        .join('');
    const h2hPlayers = result.lineup.filter(e => e.h2h && e.h2h.matches > 0);
    const h2hHtml = h2hPlayers.length
        ? `<div class="xi-h2h"><b>H2H VS ${oppName.toUpperCase()}</b> (${h2hPlayers[0].h2h.matches} ${h2hPlayers[0].h2h.matches === 1 ? 'meeting' : 'meetings'} this season)<br>${h2hPlayers.map(e =>
            `${e.name}: ${e.h2h.minutes} min · ${e.h2h.goals}G ${e.h2h.assists}A · xG ${e.h2h.xg.toFixed(2)}`).join('<br>')}</div>`
        : '';
    const subsHtml = (result.subs && result.subs.length)
        ? `<div class="xi-subs-title">SUGGESTED SUBS (${result.subs.length})</div>${result.subs.map(s => `
            <div class="sub-row">
                <span class="sub-out">${s.out}</span>
                <span class="sub-arrow">→</span>
                <b class="sub-in">${s.in}</b>
                <span class="sub-delta ${s.ratingDelta >= 0 ? 'up' : ''}">${s.ratingDelta >= 0 ? '+' : ''}${s.ratingDelta}</span>
                <div class="sub-reason">${s.reason}</div>
            </div>`).join('')}`
        : '';
    const notesHtml = result.notes && result.notes.length
        ? `<div class="xi-notes">${result.notes.map(n => `<div>• ${n}</div>`).join('')}</div>` : '';
    return `
        <div class="analysis-card">
            <div class="card-title">${label} <span class="accent-text" style="font-weight:400;">· ${result.formation}</span></div>
            <div class="card-body">
                <div class="xi-grid">${slots}</div>
                ${h2hHtml}
                ${subsHtml}
                ${notesHtml}
            </div>
        </div>
    `;
}

// ─────────────────────────────────────────────────────────────────────────────
// Contribute Tab — Submission Functions
// ─────────────────────────────────────────────────────────────────────────────

function toggleSection(headerEl) {
    const body = headerEl.nextElementSibling;
    const isOpen = body.style.display !== 'none';
    body.style.display = isOpen ? 'none' : 'block';
    headerEl.textContent = (isOpen ? '▶' : '▼') + ' ' + headerEl.textContent.substring(2);
}

async function loadContributeTeams(league) {
    const selects = [
        'match-home-team', 'match-away-team',
        'tactical-home-team', 'tactical-away-team',
        'profile-edit-team'
    ];
    const query = `query GetTeams($league: String!) { leagueTeams(league: $league) }`;
    const data = await graphqlQuery(query, { league });
    const teams = data ? data.leagueTeams : [];
    selects.forEach(sid => {
        const sel = document.getElementById(sid);
        if (!sel) return;
        sel.innerHTML = '';
        teams.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = t.replace(/_/g, ' ');
            sel.appendChild(opt);
        });
    });
}

async function submitMatchRecord() {
    console.log('submitMatchRecord CALLED');
    if (!token) {
        console.log('No token — showing auth modal');
        alert("Login required.");
        showAuthModal();
        return;
    }
    const statusEl = document.getElementById('match-submit-status');
    const btn = document.getElementById('submit-match-btn');
    if (!statusEl) { console.error('match-submit-status element NOT FOUND'); return; }
    if (!btn) { console.error('submit-match-btn NOT FOUND'); return; }
    btn.disabled = true;
    btn.textContent = 'SUBMITTING...';
    statusEl.style.display = 'block';
    statusEl.innerHTML = '<div style="color:#0096ff;padding:8px;border:1px solid #0096ff">⏳ Processing...</div>';
    const get = id => { const el = document.getElementById(id); if(!el){console.error('Missing element:',id);return '';} return el.value; };
    const getInt = id => { const el = document.getElementById(id); if(!el){return 0;} const v = parseInt(el.value); return isNaN(v) ? 0 : v; };
    const getFloat = id => { const v = get(id); return v ? parseFloat(v) : null; };
    const payload = {
        home_team: get('match-home-team'), away_team: get('match-away-team'),
        match_date: get('match-submit-date'),
        league: get('match-league-select'), season: get('match-season'),
        home_goals: getInt('match-fthg'), away_goals: getInt('match-ftag'),
        home_ht_goals: getInt('match-hthg'), away_ht_goals: getInt('match-htag'),
        home_xg: getFloat('match-home-xg'), away_xg: getFloat('match-away-xg'),
        home_shots: getInt('match-hs'), away_shots: getInt('match-as'),
        home_sot: getInt('match-hst'), away_sot: getInt('match-ast'),
        home_corners: getInt('match-hc'), away_corners: getInt('match-ac'),
        home_fouls: getInt('match-hf'), away_fouls: getInt('match-af'),
        home_yellows: getInt('match-hy'), away_yellows: getInt('match-ay'),
        home_reds: getInt('match-hr'), away_reds: getInt('match-ar'),
    };
    console.log('Submitting payload:', payload);
    try {
        const res = await fetch(`${API_BASE}/submissions/match`, {
            method: 'POST', headers: {'Content-Type':'application/json','Authorization':`Bearer ${token}`},
            body: JSON.stringify(payload)
        });
        console.log('Response status:', res.status);
        if (res.ok) {
            const data = await res.json();
            console.log('Success data:', data);
            statusEl.innerHTML = `<div class="status-banner success">✓ MATCH RECORD SUBMITTED — PENDING ADMIN REVIEW<br><small>ID #${data.id} | ${data.home_team} vs ${data.away_team} | ${data.match_date}</small></div>`;
        } else {
            const text = await res.text();
            console.error('Error response:', text);
            let errMsg = text.substring(0, 200);
            try { const err = JSON.parse(text); errMsg = err.detail || errMsg; } catch (_) {}
            statusEl.innerHTML = `<div class="status-banner error">✗ ${errMsg}</div>`;
        }
    } catch(e) {
        console.error('Network error:', e);
        statusEl.innerHTML = `<div class="status-banner error">✗ Network error: ${e.message}</div>`;
    }
    btn.disabled = false;
    btn.textContent = 'SUBMIT MATCH RECORD';
}

async function submitTacticalAnalysis() {
    if (!token) { alert("Login required."); showAuthModal(); return; }
    const get = id => document.getElementById(id).value;
    const analysisText = get('tactical-analysis-text');
    const statusEl = document.getElementById('tactical-submit-status');
    const btn = document.getElementById('submit-tactical-btn');
    if (analysisText.length < 20) {
        statusEl.innerHTML = '<div class="status-banner error">✗ Analysis must be at least 20 characters</div>';
        return;
    }
    const payload = {
        home_team: get('tactical-home-team'), away_team: get('tactical-away-team'),
        match_date: get('tactical-match-date'), analysis_text: analysisText
    };
    btn.disabled = true; btn.textContent = 'SUBMITTING...';
    statusEl.innerHTML = '<span style="color:#0096ff">⏳ Processing...</span>';
    try {
        const res = await fetch(`${API_BASE}/submissions/tactical-analysis`, {
            method: 'POST', headers: {'Content-Type':'application/json','Authorization':`Bearer ${token}`},
            body: JSON.stringify(payload)
        });
        if (res.ok) {
            const data = await res.json();
            statusEl.innerHTML = `<div class="status-banner success">✓ TACTICAL ANALYSIS SUBMITTED — PENDING ADMIN REVIEW<br><small>ID #${data.id} | ${data.home_team} vs ${data.away_team}</small></div>`;
        } else {
            let errMsg = 'Submission failed';
            try { const err = await res.json(); errMsg = err.detail || errMsg; } catch (_) {}
            statusEl.innerHTML = `<div class="status-banner error">✗ ${errMsg}</div>`;
        }
    } catch(e) { statusEl.innerHTML = `<div class="status-banner error">✗ Network error: ${e.message}</div>`; }
    btn.disabled = false; btn.textContent = 'SUBMIT ANALYSIS';
}

async function submitTeamProfileEdit() {
    if (!token) { alert("Login required."); showAuthModal(); return; }
    const get = id => document.getElementById(id).value || null;
    const payload = {
        team_name: get('profile-edit-team'),
        suggested_attack_tactic: get('profile-attack-tactic'),
        suggested_defense_tactic: get('profile-defense-tactic'),
        suggested_attack_headline: get('profile-attack-headline'),
        suggested_defense_headline: get('profile-defense-headline'),
        suggested_strengths: get('profile-strengths'),
        suggested_weaknesses: get('profile-weaknesses'),
    };
    const hasField = (payload.suggested_attack_tactic || payload.suggested_defense_tactic
        || payload.suggested_attack_headline || payload.suggested_defense_headline
        || payload.suggested_strengths || payload.suggested_weaknesses);
    if (!hasField) {
        document.getElementById('profile-submit-status').innerHTML = '<div class="status-banner error">✗ Fill at least one tactical field</div>';
        return;
    }
    const statusEl = document.getElementById('profile-submit-status');
    const btn = document.getElementById('submit-profile-btn');
    btn.disabled = true; btn.textContent = 'SUBMITTING...';
    statusEl.innerHTML = '<span style="color:#0096ff">⏳ Processing...</span>';
    try {
        const res = await fetch(`${API_BASE}/submissions/team-profile`, {
            method: 'POST', headers: {'Content-Type':'application/json','Authorization':`Bearer ${token}`},
            body: JSON.stringify(payload)
        });
        if (res.ok) {
            const data = await res.json();
            statusEl.innerHTML = `<div class="status-banner success">✓ TEAM PROFILE EDIT SUBMITTED — PENDING ADMIN REVIEW<br><small>ID #${data.id} | Team: ${data.team_name}</small></div>`;
        } else {
            let errMsg = 'Submission failed';
            try { const err = await res.json(); errMsg = err.detail || errMsg; } catch (_) {}
            statusEl.innerHTML = `<div class="status-banner error">✗ ${errMsg}</div>`;
        }
    } catch(e) { statusEl.innerHTML = `<div class="status-banner error">✗ Network error: ${e.message}</div>`; }
    btn.disabled = false; btn.textContent = 'SUBMIT PROFILE EDIT';
}

// ─────────────────────────────────────────────────────────────────────────────
// Supervisor Queue
// ─────────────────────────────────────────────────────────────────────────────

async function fetchSupervisorQueue() {
    if (!token) return;
    const container = document.getElementById('queue-container');
    const countEl = document.getElementById('queue-count');
    container.innerHTML = '<div class="empty-state"><div class="empty-icon">⏳</div><div>LOADING QUEUE...</div></div>';
    try {
        const res = await fetch(`${API_BASE}/supervisor/submissions`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        if (!res.ok) {
            const errText = await res.text();
            const msg = res.status === 403 ? 'PERMISSION DENIED — Supervisor role required'
                       : res.status === 401 ? 'SESSION EXPIRED — Please login again'
                       : `SERVER ERROR ${res.status}: ${errText.substring(0, 100)}`;
            container.innerHTML = `<div class="empty-state" style="color:#ff3333"><div class="empty-icon">❌</div><div>${msg}</div></div>`;
            return;
        }
        const items = await res.json();
        countEl.textContent = `${items.length} ITEM${items.length !== 1 ? 'S' : ''}`;
        if (items.length === 0) {
            container.innerHTML = '<div class="empty-state"><div class="empty-icon">⚡</div><div>NO PENDING SUBMISSIONS</div><div class="empty-desc">All user submissions have been reviewed.</div></div>';
            return;
        }
        renderQueueCards(items);
    } catch(e) { container.innerHTML = `<div class="empty-state" style="color:#ff3333"><div class="empty-icon">❌</div><div>ERROR</div><div class="empty-desc">${e.message}</div></div>`; }
}

function renderQueueCards(items) {
    const container = document.getElementById('queue-container');
    container.innerHTML = '';
    items.forEach(item => {
        const typeLabel = item.type === 'match_submission' ? 'MATCH RECORD'
            : item.type === 'tactical_analysis' ? 'TACTICAL ANALYSIS' : 'TEAM PROFILE';
        const typeClass = item.type === 'match_submission' ? 'match'
            : item.type === 'tactical_analysis' ? 'tactical' : 'profile';
        const detailsHtml = Object.entries(item.details || {}).filter(([,v]) => v != null).map(([k,v]) => `<div class="detail-row"><span class="detail-key">${k.replace(/_/g,' ').toUpperCase()}</span><span class="detail-val">${String(v).substring(0,120)}</span></div>`).join('');

        const card = document.createElement('div');
        card.className = 'queue-card';
        card.id = `queue-card-${item.id}`;
        card.innerHTML = `
            <div class="card-header">
                <span class="type-badge ${typeClass}">${typeLabel}</span>
                <span class="card-summary">${item.summary}</span>
                <span class="card-by">by <b>${item.username}</b></span>
            </div>
            <div class="card-details">${detailsHtml}</div>
            <div class="review-actions">
                <textarea class="tech-textarea admin-notes" placeholder="Admin notes (optional)..." rows="2"></textarea>
                <div class="review-btns">
                    <button class="tech-btn approve-btn" data-id="${item.id}" data-type="${item.type}">✓ APPROVE</button>
                    <button class="tech-btn reject-btn" data-id="${item.id}" data-type="${item.type}">✗ REJECT</button>
                </div>
            </div>
        `;
        container.appendChild(card);

        card.querySelector('.approve-btn').addEventListener('click', () => handleReview(item.id, item.type, 'approved', card));
        card.querySelector('.reject-btn').addEventListener('click', () => handleReview(item.id, item.type, 'rejected', card));
    });
}

async function handleReview(submissionId, submissionType, status, cardEl) {
    const notesTa = cardEl.querySelector('.admin-notes');
    const adminNotes = notesTa ? notesTa.value || null : null;
    try {
        const res = await fetch(`${API_BASE}/supervisor/submissions/${submissionId}/review?type=${submissionType}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
            body: JSON.stringify({ status, admin_notes: adminNotes })
        });
        if (res.ok) {
            cardEl.style.opacity = '0';
            setTimeout(() => {
                cardEl.remove();
                const remaining = document.querySelectorAll('.queue-card').length;
                document.getElementById('queue-count').textContent = `${remaining} ITEM${remaining !== 1 ? 'S' : ''}`;
                if (remaining === 0) {
                    document.getElementById('queue-container').innerHTML = '<div class="empty-state"><div class="empty-icon">⚡</div><div>NO PENDING SUBMISSIONS</div><div class="empty-desc">All user submissions have been reviewed.</div></div>';
                }
            }, 300);
        } else {
            const err = await res.json();
            const errDiv = document.createElement('div');
            errDiv.className = 'review-error';
            errDiv.textContent = `Error: ${err.detail || 'Review failed'}`;
            cardEl.appendChild(errDiv);
        }
    } catch(e) {
        const errDiv = document.createElement('div');
        errDiv.className = 'review-error';
        errDiv.textContent = `Error: ${e.message}`;
        cardEl.appendChild(errDiv);
    }
}
