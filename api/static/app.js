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
    initTabNavigation();
    checkAuthStatus();
    loadLeagueTeams(selectedLeague);
    setupEventListeners();
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
        });
    });
}

// Authentication & Token management
async function checkAuthStatus() {
    const authSection = document.getElementById('auth-section');
    const userDisplay = document.getElementById('user-display');
    const quickLoginBtn = document.getElementById('quick-login-btn');
    
    if (token) {
        try {
            const res = await fetch(`${API_BASE}/auth/me`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.ok) {
                const user = await res.json();
                userDisplay.textContent = `USER: ${user.username.toUpperCase()} (${user.role.toUpperCase()})`;
                quickLoginBtn.textContent = 'LOGOUT';
                quickLoginBtn.classList.remove('accent-border');
                quickLoginBtn.classList.add('muted-btn');
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
    quickLoginBtn.textContent = 'QUICK LOGIN';
    quickLoginBtn.classList.remove('muted-btn');
    quickLoginBtn.classList.add('accent-border');
}

async function performQuickLogin() {
    const quickLoginBtn = document.getElementById('quick-login-btn');
    if (token) {
        // Logout action
        localStorage.removeItem('access_token');
        token = null;
        checkAuthStatus();
        return;
    }
    
    quickLoginBtn.textContent = 'AUTHENTICATING...';
    
    // Call OAuth2 standard form-data login with default credentials
    const formData = new URLSearchParams();
    formData.append('username', 'admin');
    formData.append('password', 'AdminPass123!');
    
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
            // Reload context
            loadLeagueTeams(selectedLeague);
        } else {
            alert('Failed to authenticate with default supervisor credentials.');
            quickLoginBtn.textContent = 'QUICK LOGIN';
        }
    } catch (e) {
        console.error(e);
        quickLoginBtn.textContent = 'QUICK LOGIN';
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
    // Quick login button
    document.getElementById('quick-login-btn').addEventListener('click', performQuickLogin);
    
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
        const verdict = data.predicted_result === 'H' ? 'HOME WIN' : (data.predicted_result === 'A' ? 'AWAY WIN' : 'DRAW');
        
        // Parse the LLM response if it is structured JSON
        let analysis = {
            prediction_verdict: data.tactical_analysis,
            confidence_rating: "Medium",
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
                            <div class="prob-bar-fill" style="width: ${data.predicted_result === 'H' ? '65%' : '15%'}"></div>
                        </div>
                        <span class="prob-val">${data.predicted_result === 'H' ? '65%' : '15%'}</span>
                    </div>
                    <div class="prob-row">
                        <span class="prob-label">DRAW</span>
                        <div class="prob-bar-wrapper">
                            <div class="prob-bar-fill" style="width: ${data.predicted_result === 'D' ? '60%' : '20%'}"></div>
                        </div>
                        <span class="prob-val">${data.predicted_result === 'D' ? '60%' : '20%'}</span>
                    </div>
                    <div class="prob-row">
                        <span class="prob-label">AWAY</span>
                        <div class="prob-bar-wrapper">
                            <div class="prob-bar-fill" style="width: ${data.predicted_result === 'A' ? '65%' : '15%'}"></div>
                        </div>
                        <span class="prob-val">${data.predicted_result === 'A' ? '65%' : '15%'}</span>
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
    const content = input.value.trim();
    if (!content || !activeConversationId) return;
    
    input.value = '';
    
    // Add user message to UI immediately for feedback
    const messagesContainer = document.getElementById('chat-messages-container');
    const userDiv = document.createElement('div');
    userDiv.className = 'chat-msg user';
    userDiv.textContent = content;
    messagesContainer.appendChild(userDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Add temporary loading indicator
    const loaderDiv = document.createElement('div');
    loaderDiv.className = 'chat-msg assistant';
    loaderDiv.innerHTML = '<span class="accent-text">Analyzing database and streaming response...</span>';
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
            // Remove loader and insert actual response
            loaderDiv.textContent = data.content;
        } else {
            loaderDiv.textContent = "[Error: Failed to fetch response]";
        }
    } catch (e) {
        console.error(e);
        loaderDiv.textContent = `[Error: ${e.message}]`;
    }
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
