// background.js

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'CHECK_HALLUCINATION') {
        const mode = message.payload.mode || 'gemini';
        console.log(`[BACKGROUND] Received data for mode: ${mode}`, message.payload);

        // Set loading state for specific mode
        const loadingKey = `isAnalyzing_${mode}`;
        const storageUpdate = {};
        storageUpdate[loadingKey] = true;

        chrome.storage.local.set(storageUpdate, () => {
            console.log(`[BACKGROUND] Analysis started for ${mode} - loader should be visible`);
        });

        // Call Backend API
        console.log('[BACKGROUND] Calling backend API at https://lunexa-chrome-extension.onrender.com/score');
        fetch('https://lunexa-chrome-extension.onrender.com/score', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(message.payload)
        })
            .then(response => {
                console.log('[BACKGROUND] Got response from backend, status:', response.status);
                return response.json();
            })
            .then(data => {
                console.log('[BACKGROUND] Backend data:', data);
                const result = {
                    ...data,
                    timestamp: new Date().toISOString(),
                    query: message.payload.query,
                    response: message.payload.response
                };

                // Save to local storage with mode-specific key
                const resultKey = `result_${mode}`;
                const saveUpdate = {};
                saveUpdate[resultKey] = result;
                saveUpdate[loadingKey] = false;

                chrome.storage.local.set(saveUpdate, () => {
                    console.log(`[BACKGROUND] Result saved for ${mode}:`, result);
                    sendResponse({ success: true, data: result });
                });
            })
            .catch(error => {
                console.error('[BACKGROUND] Error fetching score:', error);

                // Set loading to false even on error
                const errorUpdate = {};
                errorUpdate[loadingKey] = false;
                errorUpdate[`error_${mode}`] = 'Failed to connect to backend server. Is it running?';

                chrome.storage.local.set(errorUpdate, () => {
                    console.error('[BACKGROUND] Error saved to storage');
                    sendResponse({ success: false, error: error.message });
                });
            });
    } else if (message.type === 'OPEN_POPUP') {
        // Open the extension popup
        chrome.action.openPopup();
    }
    return true;
});
