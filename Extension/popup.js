// popup.js

document.addEventListener('DOMContentLoaded', () => {
    const issuesListEl = document.getElementById('issues-list');
    const issueCountEl = document.getElementById('issue-count');
    // resultContainer removed as we use specific mode containers
    const noDataEl = document.getElementById('no-data');
    const timestampEl = document.getElementById('timestamp');
    const loaderEl = document.getElementById('loader');

    // Mode elements
    const modeBtns = document.querySelectorAll('.mode-btn');
    const instructions = document.querySelectorAll('.instruction');
    const analyzeArticleBtn = document.getElementById('analyze-article-btn');
    const analyzeSelectionBtn = document.getElementById('analyze-selection-btn');

    let currentMode = 'gemini';

    // SVG gradient is now defined in HTML, no need to create it here

    function showLoader() {
        loaderEl.classList.remove('hidden');
        document.querySelectorAll('.result-view').forEach(el => el.classList.add('hidden'));
        noDataEl.classList.add('hidden');
    }

    function hideLoader() {
        loaderEl.classList.add('hidden');
    }

    // Mode switching
    function switchMode(mode) {
        currentMode = mode;

        // Update button states
        modeBtns.forEach(btn => {
            if (btn.dataset.mode === mode) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Update instructions
        instructions.forEach(instruction => {
            if (instruction.id === `instruction-${mode}`) {
                instruction.classList.add('active');
            } else {
                instruction.classList.remove('active');
            }
        });

        // Hide all result containers
        document.querySelectorAll('.result-view').forEach(el => el.classList.add('hidden'));

        // Show current mode result container (if we have data)
        // We'll let loadAndDisplayData handle showing the container or empty state
        loadAndDisplayData();

        // Save mode preference
        chrome.storage.local.set({ currentMode: mode });
    }

    // Mode button click handlers
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            switchMode(btn.dataset.mode);
        });
    });

    // Analyze Article button
    analyzeArticleBtn.addEventListener('click', () => {
        showLoader();
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (!tabs[0]) {
                hideLoader();
                alert('No active tab found');
                return;
            }

            chrome.tabs.sendMessage(tabs[0].id, { type: 'ANALYZE_ARTICLE' }, (response) => {
                if (chrome.runtime.lastError) {
                    console.error('Error:', chrome.runtime.lastError);
                    hideLoader();
                    // More helpful error message
                    alert('Extension not ready on this page. Please reload the page and try again.');
                }
            });
        });
    });

    // Analyze Selection button
    analyzeSelectionBtn.addEventListener('click', () => {
        chrome.storage.local.get(['selectedText'], (result) => {
            if (!result.selectedText) {
                alert('Please select some text on the page first.');
                return;
            }

            showLoader();

            // Send analysis request
            const data = {
                query: 'Text Selection Analysis',
                response: result.selectedText.substring(0, 5000),
                mode: 'selection'
            };

            if (!chrome.runtime?.id) {
                alert('Extension context invalidated. Please close and reopen the popup.');
                return;
            }

            chrome.runtime.sendMessage({
                type: "CHECK_HALLUCINATION",
                payload: data
            }, (response) => {
                if (chrome.runtime.lastError) {
                    console.error('Error:', chrome.runtime.lastError);
                    hideLoader();
                }
            });
        });
    });

    // Load saved mode
    chrome.storage.local.get(['currentMode'], (result) => {
        if (result.currentMode) {
            switchMode(result.currentMode);
        }
    });

    // Selection mode elements
    const selectedTextPreview = document.getElementById('selected-text-preview');
    const selectionCharCount = document.getElementById('selection-char-count');
    const selectionWordCount = document.getElementById('selection-word-count');

    // Update selected text preview
    function updateSelectionPreview() {
        chrome.storage.local.get(['selectedText'], (result) => {
            if (result.selectedText) {
                const text = result.selectedText;
                selectedTextPreview.textContent = text.length > 150 ? text.substring(0, 150) + '...' : text;
                selectionCharCount.textContent = text.length;
                selectionWordCount.textContent = text.split(/\s+/).filter(w => w.length > 0).length;
            } else {
                selectedTextPreview.textContent = 'No text selected yet';
                selectionCharCount.textContent = '0';
                selectionWordCount.textContent = '0';
            }
        });
    }

    // Update preview when storage changes
    chrome.storage.onChanged.addListener((changes, namespace) => {
        if (namespace === 'local' && changes.selectedText) {
            updateSelectionPreview();
        }
    });

    // Initial preview update
    updateSelectionPreview();

    function loadAndDisplayData() {
        const mode = currentMode;
        const resultKey = `result_${mode}`;
        const loadingKey = `isAnalyzing_${mode}`;

        console.log('[POPUP] Loading data for mode:', mode);
        console.log('[POPUP] Looking for keys:', resultKey, loadingKey);

        chrome.storage.local.get([resultKey, loadingKey], (result) => {
            console.log('[POPUP] Storage result:', result);

            const isAnalyzing = result[loadingKey];
            const data = result[resultKey];

            console.log('[POPUP] isAnalyzing:', isAnalyzing);
            console.log('[POPUP] data:', data);

            // If currently analyzing, show loader
            if (isAnalyzing) {
                console.log('[POPUP] Showing loader');
                showLoader();
                return;
            }

            // If we have results and not analyzing, display them
            if (data) {
                console.log('[POPUP] Displaying results:', data);
                displayResults(data, mode);
            } else {
                console.log('[POPUP] No data, showing empty state');
                showEmptyState();
            }
        });
    }

    function displayResults(data, mode) {
        console.log('[POPUP] displayResults called with:', data, mode);

        const scores = data.scores || {};
        console.log('[POPUP] scores:', scores);

        const containerId = `result-${mode}`;
        const container = document.getElementById(containerId);

        console.log('[POPUP] container:', containerId, container);

        if (!container) {
            console.error('[POPUP] Container not found!', containerId);
            return;
        }

        // Format score values
        const formatScore = (val) => {
            if (val === undefined || val === null) return '--';
            return val.toFixed(2) + '%';
        };

        // Helper to safely set text content
        const setText = (selector, text) => {
            const el = container.querySelector(selector);
            if (el) el.textContent = text;
        };

        // Display CARS score with animation
        const carsScore = scores.CARS || 0;
        console.log('[POPUP] CARS score:', carsScore);
        setText(`.${mode}-cars-score`, formatScore(carsScore));

        // Animate score ring
        const circumference = 2 * Math.PI * 54; // radius is 54
        const scoreRingFill = container.querySelector(`.${mode}-score-ring`);

        // Normalize CARS score to 0-100 range for visual display
        let normalizedScore = 0;
        if (carsScore >= 100) {
            normalizedScore = 100;
        } else if (carsScore <= -100) {
            normalizedScore = 0;
        } else {
            normalizedScore = ((carsScore + 100) / 200) * 100;
        }

        const offset = circumference - (normalizedScore / 100) * circumference;

        if (scoreRingFill) {
            setTimeout(() => {
                scoreRingFill.style.strokeDashoffset = offset;
            }, 100);
        }

        // Update score status
        const scoreStatus = container.querySelector(`.${mode}-score-status`);
        if (scoreStatus) {
            if (carsScore >= 8) {
                scoreStatus.textContent = 'Excellent Reliability';
                scoreStatus.className = `score-status ${mode}-score-status excellent`;
            } else if (carsScore >= 5) {
                scoreStatus.textContent = 'Good Reliability';
                scoreStatus.className = `score-status ${mode}-score-status good`;
            } else if (carsScore >= 0) {
                scoreStatus.textContent = 'Moderate Concerns';
                scoreStatus.className = `score-status ${mode}-score-status warning`;
            } else {
                scoreStatus.textContent = 'High Risk Detected';
                scoreStatus.className = `score-status ${mode}-score-status danger`;
            }
        }

        // Update all metrics with percentage
        setText(`.${mode}-factual-accuracy`, formatScore(scores.Factual_Accuracy));
        setText(`.${mode}-reasoning-integrity`, formatScore(scores.Reasoning_Integrity));
        setText(`.${mode}-evidence-alignment`, formatScore(scores.Evidence_Alignment));
        setText(`.${mode}-consistency`, formatScore(scores.Consistency));
        setText(`.${mode}-fake-news`, formatScore(scores.Fake_News_Likelihood));
        setText(`.${mode}-trust-confidence`, formatScore(scores.Trust_Confidence_Score));
        setText(`.${mode}-hallucination-prob`, formatScore(scores.Hallucination_Probability));

        // Display query and response
        const userQueryText = data.query || 'N/A';
        const responseText = data.response || 'N/A';

        setText(`.${mode}-user-query`, userQueryText);
        setText(`.${mode}-response`, responseText);

        // Show results, hide empty state and loader
        hideLoader();

        // Hide all result containers first
        document.querySelectorAll('.result-view').forEach(el => el.classList.add('hidden'));

        // Show current container
        container.classList.remove('hidden');
        noDataEl.classList.add('hidden');
    }

    function showEmptyState() {
        console.log('[POPUP] showEmptyState called');
        hideLoader();
        document.querySelectorAll('.result-view').forEach(el => el.classList.add('hidden'));
        noDataEl.classList.remove('hidden');
    }

    // Initial load - check if analyzing
    loadAndDisplayData();

    // Listen for storage changes to update in real-time
    chrome.storage.onChanged.addListener((changes, namespace) => {
        if (namespace === 'local') {
            const mode = currentMode;
            const resultKey = `result_${mode}`;
            const loadingKey = `isAnalyzing_${mode}`;

            // If analyzing state changed or results changed for CURRENT mode, reload
            if (changes[loadingKey] || changes[resultKey]) {
                loadAndDisplayData();
            }
        }
    });

    // Poll for updates every 500ms for smoother loader experience
    setInterval(() => {
        loadAndDisplayData();
    }, 500);
});
