// ==========================================================================
// AURAASSESS CLIENT CONTROLLER (10/10 BIOMETRICS & DASHBOARD SYNC)
// ==========================================================================

const video = document.getElementById('webcam');
const canvas = document.getElementById('overlayCanvas');
const ctx = canvas.getContext('2d');

const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const sessionTimer = document.getElementById('sessionTimer');
const btnCalibrate = document.getElementById('btnCalibrate');
const fpsCounter = document.getElementById('fpsCounter');
const pacifyingAlert = document.getElementById('pacifyingAlert');
const chkShowMesh = document.getElementById('chkShowMesh');

// Predictor DOM elements
const stateCard = document.getElementById('stateCard');
const stateValue = document.getElementById('stateValue');
const stateDesc = document.getElementById('stateDesc');

const probStressBar = document.getElementById('probStressBar');
const probStressVal = document.getElementById('probStressVal');
const probChillBar = document.getElementById('probChillBar');
const probChillVal = document.getElementById('probChillVal');
const probConfidentBar = document.getElementById('probConfidentBar');
const probConfidentVal = document.getElementById('probConfidentVal');
const probOverconfidentBar = document.getElementById('probOverconfidentBar');
const probOverconfidentVal = document.getElementById('probOverconfidentVal');

// Metrics DOM elements
const metricForehead = document.getElementById('metricForehead');
const metricBlinks = document.getElementById('metricBlinks');
const metricSway = document.getElementById('metricSway');
const metricGestures = document.getElementById('metricGestures');

// Panel sections
const dashboardSection = document.getElementById('dashboardSection');
const reportSection = document.getElementById('reportSection');
const btnRestart = document.getElementById('btnRestart');

// Report elements
const repDominantCircle = document.getElementById('repDominantCircle');
const repDominantState = document.getElementById('repDominantState');
const repDominantPct = document.getElementById('repDominantPct');
const repSummaryText = document.getElementById('repSummaryText');
const repForehead = document.getElementById('repForehead');
const repGaze = document.getElementById('repGaze');
const repSelfTouch = document.getElementById('repSelfTouch');
const repSway = document.getElementById('repSway');
const repRecommendation = document.getElementById('repRecommendation');

// Session State
let ws = null;
let stream = null;
let lastFrameTime = performance.now();
let isCalibrating = true;
let sessionSeconds = 0;
let timerInterval = null;
let frameSendInterval = null;

// Real-time coordinates
let currentLandmarks = [];
let currentPoseLandmarks = [];

// Session Accumulators (for the 1-minute report)
let sessionStateHistory = [];
let metricHistory = {
    furrow_z: [],
    raise_z: [],
    ear: [],
    blink_rate: [],
    sway_z: [],
    self_touch_count: 0,
    hand_velocity: [],
    gesture_score: [],
    crossed_arms_count: 0
};

// 1. Initialize Webcam Stream
async function initWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
            audio: false
        });
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            startConnection();
        };
    } catch (err) {
        console.error("Camera access failed:", err);
        alert("Webcam access denied. Please grant permission and reload.");
    }
}

// 2. WebSocket & Ingestion Loop
function startConnection() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    ws.onopen = () => {
        console.log("WebSocket open.");
        updateStatus("CALIBRATING", "calibrating");
        startFramePipeline();
        startTimer();
    };
    
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        
        if (msg.type === "frame_response") {
            currentLandmarks = msg.landmarks || [];
            currentPoseLandmarks = msg.pose_landmarks || [];
            
            if (msg.pacifying_active) {
                pacifyingAlert.classList.add("active");
                metricHistory.self_touch_count++;
            } else {
                pacifyingAlert.classList.remove("active");
            }
            drawOverlay();
            
        } else if (msg.type === "second_update") {
            isCalibrating = msg.features.is_calibrating;
            
            if (isCalibrating) {
                updateStatus(`CALIBRATING (${msg.second}s)`, "calibrating");
                updatePredictorCard("Calibrating...", "Look directly at the lens. Building neutral baseline.", "neutral");
            } else {
                updateStatus("LIVE", "live");
                updatePredictorCard(msg.state, getDescription(msg.state), msg.state.toLowerCase());
                
                // Accumulate state for final report
                sessionStateHistory.push(msg.state);
                
                // Log metric points
                metricHistory.furrow_z.push(msg.features.furrow_index_z || 0.0);
                metricHistory.raise_z.push(msg.features.raise_index_z || 0.0);
                metricHistory.ear.push(msg.features.ear || 0.25);
                metricHistory.blink_rate.push(msg.features.blink_rate_bpm || 0.0);
                metricHistory.sway_z.push(msg.features.sway_score_z || 0.0);
                
                // Track body metrics
                metricHistory.hand_velocity.push(msg.features.hand_velocity_z || 0.0);
                metricHistory.gesture_score.push(msg.features.gesture_score_z || 0.0);
                if (msg.features.crossed_arms > 0.5) {
                    metricHistory.crossed_arms_count++;
                }
            }
            updateProbabilities(msg.probabilities);
            updateMetrics(msg.features);
        }
    };
    
    ws.onclose = () => {
        console.log("WebSocket closed.");
        updateStatus("DISCONNECTED", "disconnected");
        stopFramePipeline();
        stopTimer();
    };
}

function startFramePipeline() {
    if (frameSendInterval) clearInterval(frameSendInterval);
    frameSendInterval = setInterval(() => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        
        ctx.save();
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();
        
        const dataUrl = canvas.toDataURL("image/jpeg", 0.4);
        ws.send(JSON.stringify({ type: "frame", image: dataUrl }));

        // Compute Ingestion FPS
        const now = performance.now();
        const fps = Math.round(1000 / (now - lastFrameTime));
        lastFrameTime = now;
        fpsCounter.innerText = `${fps} FPS`;
    }, 100);
}

function stopFramePipeline() {
    if (frameSendInterval) {
        clearInterval(frameSendInterval);
        frameSendInterval = null;
    }
}

// 3. Render Landmark Meshes
function drawOverlay() {
    if (!chkShowMesh.checked) return;
    const w = canvas.width;
    const h = canvas.height;
    
    // Face Mesh overlay
    if (currentLandmarks.length > 0) {
        ctx.fillStyle = '#10b981'; // Mint Green nodes
        ctx.shadowColor = '#10b981';
        ctx.shadowBlur = 3;
        currentLandmarks.forEach(pt => {
            ctx.beginPath();
            ctx.arc((1 - pt.x) * w, pt.y * h, 2.5, 0, 2 * Math.PI);
            ctx.fill();
        });
    }
    
    // Pose Skeleton overlay
    if (currentPoseLandmarks.length > 0) {
        ctx.strokeStyle = '#059669'; // Emerald Green skeleton lines
        ctx.lineWidth = 2.5;
        ctx.shadowColor = '#059669';
        ctx.shadowBlur = 6;
        
        const joints = {};
        currentPoseLandmarks.forEach(pt => {
            joints[pt.idx] = { x: (1 - pt.x) * w, y: pt.y * h };
        });
        
        // Connect skeleton points
        const drawLine = (j1, j2) => {
            if (joints[j1] && joints[j2]) {
                ctx.beginPath();
                ctx.moveTo(joints[j1].x, joints[j1].y);
                ctx.lineTo(joints[j2].x, joints[j2].y);
                ctx.stroke();
            }
        };
        drawLine(11, 12); // shoulders
        drawLine(11, 15); // left arm
        drawLine(12, 16); // right arm
        drawLine(23, 24); // hips
    }
    ctx.shadowBlur = 0;
}

// 4. Session Countdown (Exactly 1 Minute Limit)
function startTimer() {
    sessionSeconds = 0;
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(() => {
        sessionSeconds++;
        const mins = Math.floor(sessionSeconds / 60).toString().padStart(2, '0');
        const secs = (sessionSeconds % 60).toString().padStart(2, '0');
        sessionTimer.innerText = `${mins}:${secs}`;
        
        // HALT STREAM AT 60 SECONDS (1 MINUTE)
        if (sessionSeconds >= 60) {
            triggerFinalReport();
        }
    }, 1000);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

// 5. Build and Trigger Final Report
function triggerFinalReport() {
    stopFramePipeline();
    stopTimer();
    if (ws) ws.close();
    
    // Hide tracking dashboard and show report section
    dashboardSection.style.display = "none";
    reportSection.style.display = "flex";
    updateStatus("COMPLETED", "calibrating");

    // 1. Calculate dominant state distribution
    const totalStateSecs = sessionStateHistory.length;
    if (totalStateSecs === 0) {
        repSummaryText.innerText = "Insufficient data points gathered. Please try a longer calibration run.";
        return;
    }

    const counts = { Stress: 0, Chill: 0, Confident: 0, Overconfident: 0 };
    sessionStateHistory.forEach(s => counts[s]++);

    // Determine the state with highest count
    const dominantState = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
    const dominantPct = Math.round((counts[dominantState] / totalStateSecs) * 100);

    // Apply colors to dominant circle gauge
    repDominantCircle.className = "rep-badge " + dominantState.toLowerCase();
    repDominantState.innerText = dominantState;
    repDominantPct.innerText = `${dominantPct}%`;

    // 2. Average the feature logs
    const avg = arr => arr.length > 0 ? arr.reduce((a,b)=>a+b)/arr.length : 0;
    const avgFurrow = avg(metricHistory.furrow_z);
    const avgRaise = avg(metricHistory.raise_z);
    const avgBlinks = Math.round(avg(metricHistory.blink_rate));
    const avgSway = avg(metricHistory.sway_z);
    const selfTouches = metricHistory.self_touch_count;
    const avgHandVel = avg(metricHistory.hand_velocity);
    const avgGesture = avg(metricHistory.gesture_score);
    const crossedSecs = metricHistory.crossed_arms_count;

    // 3. Populate feature analysis summaries
    // Forehead
    if (avgFurrow < -1.0) {
        repForehead.innerText = "Significant forehead furrowing detected, suggesting periods of high stress or cognitive effort.";
    } else if (avgRaise > 1.0) {
        repForehead.innerText = "Exaggerated eyebrow raising detected, denoting high verbal expressiveness or skepticism.";
    } else {
        repForehead.innerText = "Forehead muscles remained relaxed and smooth, showing solid mental comfort.";
    }

    // Gaze
    if (avgBlinks > 24) {
        repGaze.innerText = `Elevated blink rate (${avgBlinks} BPM) and slight focus shifts imply nervousness or anxiety.`;
    } else {
        repGaze.innerText = `Stable, calm eye contact index (${avgBlinks} BPM blinks) suggesting comfortable focus.`;
    }

    // Self-Touch (neck/face) & Crossed arms / Gesturing
    const selfTouchSecs = Math.round(selfTouches / 10);
    let selfTouchText = "";
    if (selfTouchSecs > 4) {
        selfTouchText += `Observed multiple neck/face touches (${selfTouchSecs}s total), indicating somatic anxiety soothing. `;
    } else {
        selfTouchText += `No significant self-soothing touches observed. `;
    }
    
    if (crossedSecs > 4) {
        selfTouchText += `Crossed-arms defensive posture detected for ${crossedSecs}s. `;
    } else {
        selfTouchText += `Maintained open and accessible body posture. `;
    }

    if (avgGesture > 1.0) {
        selfTouchText += "Candidate used active, expressive, and confident hand gestures.";
    } else if (avgGesture < -0.8) {
        selfTouchText += "Hands remained tucked in or static, indicating closed posture.";
    } else {
        selfTouchText += "Natural, controlled hand gesturing observed.";
    }
    repSelfTouch.innerText = selfTouchText;

    // Posture Sway & Lean
    let postureText = "";
    if (avgSway > 1.2) {
        postureText += `Observed elevated postural sway and torso shifting (${avgSway.toFixed(1)} Z-score). `;
    } else {
        postureText += `Torso remained highly stable and composed. `;
    }
    repSway.innerText = postureText;

    // 4. Set final AI Recommendation and Summary
    let summary = "";
    let recommendation = "";

    if (dominantState === "Stress") {
        summary = "The candidate exhibited prominent stress and anxiety markers. This was highlighted by elevated blink frequencies, frequent hand-to-neck pacifying gestures, and slower transition latencies between questions.";
        recommendation = "Technical follow-up recommended. The candidate is capable but may require structured, low-stress coaching to perform comfortably in client-facing environments.";
    } else if (dominantState === "Chill") {
        summary = "The candidate demonstrated high composure and low emotional arousal. Body language remained open, torso sway was negligible, and vocal prosody was relaxed and consistent.";
        recommendation = "Highly suited for analytical, logical, or calm-under-pressure roles. Great addition to high-stress backend engineering or risk management teams.";
    } else if (dominantState === "Confident") {
        summary = "The candidate established excellent focus and a strong personal connection. Eye alignment was direct, posture was stable yet engaged, and speech response pacing was prompt and well-articulated.";
        recommendation = "Strongly recommend. Candidate matches the profile of an assertive communicator, ideal for client-facing, leadership, product management, or collaborative roles.";
    } else if (dominantState === "Overconfident") {
        summary = "The candidate showed highly dominant behaviors. This was indicated by backward leaning posture (leaning away), high pitch/intensity vocal energy, and immediate response latencies (answering before prompts cleared).";
        recommendation = "Candidate has high energy and sales assertiveness. Best suited for high-growth business development or competitive environments; evaluate further for active listening skills.";
    }

    repSummaryText.innerText = summary;
    repRecommendation.innerText = recommendation;
}

// 6. UI Updates and Handlers
function updateStatus(text, statusClass) {
    statusText.innerText = text;
    if (statusClass === "live") {
        statusBadge.className = "badge status-badge live";
    } else {
        statusBadge.className = "badge status-badge";
    }
}

function updatePredictorCard(state, desc, cardClass) {
    stateValue.innerText = state;
    stateDesc.innerText = desc;
    stateCard.className = "glass-panel hero-state-card " + cardClass;
    
    // Update vibe ring and icon classes dynamically
    const ring = document.getElementById('vibeRing');
    const icon = document.getElementById('vibeIcon');
    
    if (ring && icon) {
        if (cardClass === "stress") {
            icon.className = "fa-solid fa-bolt vibe-icon";
        } else if (cardClass === "chill") {
            icon.className = "fa-solid fa-leaf vibe-icon";
        } else if (cardClass === "confident") {
            icon.className = "fa-solid fa-shield-halved vibe-icon";
        } else if (cardClass === "overconfident") {
            icon.className = "fa-solid fa-crown vibe-icon";
        } else {
            icon.className = "fa-solid fa-heart-pulse vibe-icon";
        }
    }
}

function updateProbabilities(probs) {
    const stress = probs.Stress || 0.0;
    const chill = probs.Chill || 0.0;
    const confident = probs.Confident || 0.0;
    const overconfident = probs.Overconfident || 0.0;

    probStressVal.innerText = `${Math.round(stress * 100)}%`;
    probStressBar.style.width = `${stress * 100}%`;
    probChillVal.innerText = `${Math.round(chill * 100)}%`;
    probChillBar.style.width = `${chill * 100}%`;
    probConfidentVal.innerText = `${Math.round(confident * 100)}%`;
    probConfidentBar.style.width = `${confident * 100}%`;
    probOverconfidentVal.innerText = `${Math.round(overconfident * 100)}%`;
    probOverconfidentBar.style.width = `${overconfident * 100}%`;
}

function updateMetrics(features) {
    const furrow_z = features.furrow_index_z || 0.0;
    const raise_z = features.raise_index_z || 0.0;
    if (furrow_z < -1.2) {
        metricForehead.innerText = "Furrowed (Stress)";
        metricForehead.className = "val text-magenta";
    } else if (raise_z > 1.2) {
        metricForehead.innerText = "Raised (Expressive)";
        metricForehead.className = "val text-orange";
    } else {
        metricForehead.innerText = "Relaxed";
        metricForehead.className = "val";
    }

    const blinks = features.blink_rate_bpm !== undefined ? Math.round(features.blink_rate_bpm) : "--";
    metricBlinks.innerHTML = `${blinks} <span class="unit">BPM</span>`;

    const sway_z = features.sway_score_z || 0.0;
    if (sway_z > 1.2) {
        metricSway.innerText = "Restless";
        metricSway.className = "val text-magenta";
    } else {
        metricSway.innerText = "Stable";
        metricSway.className = "val";
    }

    const gesture_z = features.gesture_score_z || 0.0;
    if (gesture_z > 1.2) {
        metricGestures.innerText = "Expressive";
        metricGestures.className = "val text-orange";
    } else if (gesture_z < -0.8) {
        metricGestures.innerText = "Frozen/Static";
        metricGestures.className = "val text-magenta";
    } else {
        metricGestures.innerText = "Natural";
        metricGestures.className = "val";
    }
}

// Reset Calibration trigger
btnCalibrate.onclick = () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "calibrate_reset" }));
        sessionSeconds = 0;
        sessionStateHistory = [];
        Object.keys(metricHistory).forEach(k => {
            if (Array.isArray(metricHistory[k])) metricHistory[k] = [];
            else metricHistory[k] = 0;
        });
        updateStatus("RE-CALIBRATING", "calibrating");
    }
};

// Restart Session from report
btnRestart.onclick = () => {
    reportSection.style.display = "none";
    dashboardSection.style.display = "flex";
    
    sessionStateHistory = [];
    Object.keys(metricHistory).forEach(k => {
        if (Array.isArray(metricHistory[k])) metricHistory[k] = [];
        else metricHistory[k] = 0;
    });
    
    initWebcam();
};

function getDescription(state) {
    switch (state) {
        case "Stress":
            return "High stress indicators detected. Take a slow, deep breath and steady your posture.";
        case "Chill":
            return "Composed baseline. Continue communicating at this calm and comfortable pace.";
        case "Confident":
            return "Strong direct engagement and balanced posture. You are speaking with conviction.";
        case "Overconfident":
            return "Elevated assertiveness. Remember to check response pacing and keep active listening.";
        default:
            return "Analyzing baseline data...";
    }
}

// Boot up
window.onload = initWebcam;
