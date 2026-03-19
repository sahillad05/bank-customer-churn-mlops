/* ═══════════════════════════════════════════════════════════
   ChurnGuard AI — Frontend Logic
   Handles: form submission, API calls, UI rendering
═══════════════════════════════════════════════════════════ */

const CIRCUMFERENCE = 2 * Math.PI * 50; // r=50 → 314.15

// ──────────────────────────────────────────────
// DOM References
// ──────────────────────────────────────────────
const form          = document.getElementById('prediction-form');
const predictBtn    = document.getElementById('predict-btn');
const btnText       = predictBtn.querySelector('.btn-text');
const btnIcon       = predictBtn.querySelector('.btn-icon');
const btnLoader     = predictBtn.querySelector('.btn-loader');

const resultsPanel  = document.getElementById('results-panel');
const ringFill      = document.getElementById('ring-fill');
const ringValue     = document.getElementById('churn-prob-display');
const churnStat     = document.getElementById('churn-prob-stat');
const retainStat    = document.getElementById('retention-prob-stat');
const predLabel     = document.getElementById('prediction-label');
const riskBadge     = document.getElementById('risk-badge');
const recText       = document.getElementById('recommendation-text');
const apiError      = document.getElementById('api-error');

const metricsContainer = document.getElementById('metrics-container');
const cmImg            = document.getElementById('confusion-matrix-img');
const cmError          = document.getElementById('cm-error');

// ──────────────────────────────────────────────
// Animation helpers
// ──────────────────────────────────────────────
function animateValue(element, start, end, duration, suffix = '') {
  const range    = end - start;
  const startTime = performance.now();

  function update(now) {
    const elapsed  = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    element.textContent = (start + range * eased).toFixed(2) + suffix;
    if (progress < 1) requestAnimationFrame(update);
  }

  requestAnimationFrame(update);
}

function setRing(probability) {
  // probability is 0–100
  const offset = CIRCUMFERENCE - (probability / 100) * CIRCUMFERENCE;
  ringFill.style.strokeDashoffset = offset;

  // Color the ring by risk level
  if (probability > 70) {
    ringFill.style.stroke = '#ef4444';
  } else if (probability >= 40) {
    ringFill.style.stroke = '#f59e0b';
  } else {
    ringFill.style.stroke = '#22c55e';
  }
}

// ──────────────────────────────────────────────
// Load model metrics on page load
// ──────────────────────────────────────────────
async function loadMetrics() {
  try {
    const res = await fetch('/metrics');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    metricsContainer.innerHTML = '';

    const labels = {
      accuracy: 'Accuracy',
      f1_score: 'F1 Score',
    };

    for (const [key, value] of Object.entries(data)) {
      const displayLabel = labels[key] || key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      const displayValue = typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : value;

      const card = document.createElement('div');
      card.className = 'metric-card';
      card.innerHTML = `
        <div class="metric-key">${displayLabel}</div>
        <div class="metric-value">${displayValue}</div>
      `;
      metricsContainer.appendChild(card);
    }
  } catch (err) {
    metricsContainer.innerHTML = `<p style="font-size:0.78rem;color:var(--text-muted);padding:0.5rem 0;">Could not load metrics. Run the DVC pipeline first.</p>`;
    console.warn('Metrics fetch failed:', err);
  }
}

// ──────────────────────────────────────────────
// Handle confusion matrix image error
// ──────────────────────────────────────────────
cmImg.addEventListener('error', () => {
  cmImg.classList.add('hidden');
  cmError.classList.remove('hidden');
});

// ──────────────────────────────────────────────
// Form submit → POST /predict
// ──────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  // ── Build payload ──
  const fd = new FormData(form);
  const payload = {
    CreditScore:     parseInt(fd.get('CreditScore'), 10),
    Geography:       fd.get('Geography'),
    Gender:          fd.get('Gender'),
    Age:             parseInt(fd.get('Age'), 10),
    Tenure:          parseInt(fd.get('Tenure'), 10),
    Balance:         parseFloat(fd.get('Balance')),
    NumOfProducts:   parseInt(fd.get('NumOfProducts'), 10),
    HasCrCard:       parseInt(fd.get('HasCrCard'), 10),
    IsActiveMember:  parseInt(fd.get('IsActiveMember'), 10),
    EstimatedSalary: parseFloat(fd.get('EstimatedSalary')),
  };

  // ── Validate ──
  for (const [key, val] of Object.entries(payload)) {
    if (val === '' || (typeof val === 'number' && isNaN(val))) {
      showApiError(`Please fill in all fields correctly. (${key} is invalid)`);
      return;
    }
  }

  // ── Loading state ──
  setLoading(true);
  hideApiError();
  resultsPanel.classList.add('hidden');

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errData.detail || `Request failed with status ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);

  } catch (err) {
    showApiError(`Prediction failed: ${err.message}`);
    resultsPanel.classList.remove('hidden');
    console.error('Prediction error:', err);
  } finally {
    setLoading(false);
  }
});

// ──────────────────────────────────────────────
// Render prediction results
// ──────────────────────────────────────────────
function renderResults(data) {
  const { churn_probability, retention_probability, prediction, risk_level, recommendation } = data;

  // Show panel first (but invisible so it can layout)
  resultsPanel.classList.remove('hidden');

  // Animate ring
  setRing(churn_probability);

  // Animate value counter
  animateValue(ringValue, 0, churn_probability, 900, '%');
  animateValue(churnStat,   0, churn_probability,   900, '%');
  animateValue(retainStat,  0, retention_probability, 900, '%');

  // Prediction label
  predLabel.textContent = prediction === 1 ? '🔴 Churn' : '🟢 Not Churn';
  predLabel.style.color = prediction === 1
    ? 'var(--risk-high)'
    : 'var(--risk-low)';

  // Risk badge
  riskBadge.className = 'risk-badge ' + risk_level.toLowerCase();
  const riskEmoji = { HIGH: '⚠️', MEDIUM: '⚡', LOW: '✅' };
  riskBadge.textContent = `${riskEmoji[risk_level] || ''} ${risk_level} RISK`;

  // Recommendation
  recText.textContent = recommendation;

  // Scroll to results smoothly
  setTimeout(() => {
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 100);
}

// ──────────────────────────────────────────────
// UI helpers
// ──────────────────────────────────────────────
function setLoading(loading) {
  predictBtn.disabled = loading;
  if (loading) {
    btnText.classList.add('hidden');
    btnIcon.classList.add('hidden');
    btnLoader.classList.remove('hidden');
  } else {
    btnText.classList.remove('hidden');
    btnIcon.classList.remove('hidden');
    btnLoader.classList.add('hidden');
  }
}

function showApiError(msg) {
  apiError.textContent = `⚠️ ${msg}`;
  apiError.classList.remove('hidden');
  resultsPanel.classList.remove('hidden');
}

function hideApiError() {
  apiError.classList.add('hidden');
}

// ──────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────
(function init() {
  // Set initial ring offset (empty)
  ringFill.style.strokeDasharray  = CIRCUMFERENCE;
  ringFill.style.strokeDashoffset = CIRCUMFERENCE;

  // Load metrics from API
  loadMetrics();
})();
