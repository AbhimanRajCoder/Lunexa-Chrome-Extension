// content.js - Simplified
console.log("[LUNEXA] Content script loaded");

// Check if we're on a Gemini page
const isGeminiPage = window.location.hostname.includes("gemini");
console.log("[LUNEXA] Is Gemini page:", isGeminiPage, "| URL:", window.location.hostname);

let lastSentQuery = "";
let lastSentResponse = "";
let checkTimeout = null;
let lastResponseLength = 0;
let stableCheckCount = 0;

function isGeminiGenerating() {
  // Only check for the stop button - most reliable indicator
  const stopButton = document.querySelector('button[aria-label*="Stop generating"]');
  return !!stopButton;
}

function getGeminiData() {
  if (isGeminiGenerating()) {
    return { error: "Gemini is still generating..." };
  }

  // --- User Query Selection (EXACT user logic) ---
  const userMsgs = [...document.querySelectorAll("div")]
    .filter(el => el.innerText?.trim() && el.innerText.length < 500 && el.querySelector("p"));

  // Use .pop() exactly like user's snippet
  const latestUser = userMsgs.pop();
  const userQuery = latestUser?.innerText?.trim() || "";

  if (!userQuery) {
    return { error: "Query is empty" };
  }

  // --- Gemini Response Selection (EXACT user logic) ---
  const geminiMsgs = [...document.querySelectorAll("div")]
    .filter(el => el.innerText?.trim() && el.innerText.length > 50);

  // Use .pop() exactly like user's snippet
  const latestResponse = geminiMsgs.pop();
  const aiResponse = latestResponse?.innerText?.trim() || "";

  if (!aiResponse) {
    return { error: "No valid AI response found" };
  }

  // Check if response is stable
  if (aiResponse.length !== lastResponseLength) {
    lastResponseLength = aiResponse.length;
    stableCheckCount = 0;
    return { error: "Response still changing" };
  } else {
    stableCheckCount++;
    if (stableCheckCount < 2) {
      return { error: `Stability check ${stableCheckCount}/2` };
    }
  }

  console.log("[LUNEXA] Data ready:", userQuery.substring(0, 30) + "...");
  return {
    query: userQuery,
    response: aiResponse,
    message: aiResponse, // Alias for backend compatibility if needed
    checkType: 'hallucination',
    mode: 'gemini'
  };
}

function sendToBackground(data) {
  if (!data) return;

  if (data.query === lastSentQuery && data.response === lastSentResponse) {
    return;
  }

  lastSentQuery = data.query;
  lastSentResponse = data.response;
  lastResponseLength = 0;
  stableCheckCount = 0;

  try {
    if (!chrome.runtime?.id) {
      console.error("[LUNEXA] Extension context invalid!");
      return;
    }

    chrome.runtime.sendMessage({
      type: "CHECK_HALLUCINATION",
      payload: data
    }, (response) => {
      if (chrome.runtime.lastError) {
        console.error("[LUNEXA] Send error:", chrome.runtime.lastError.message);
      } else {
        console.log("[LUNEXA] ✓ Message sent successfully");
      }
    });
  } catch (e) {
    console.error("[LUNEXA] Exception:", e);
  }
}

function scheduleCheck() {
  if (checkTimeout) clearTimeout(checkTimeout);

  checkTimeout = setTimeout(() => {
    if (!isGeminiPage) return;

    const result = getGeminiData();

    if (result && !result.error) {
      console.log('[LUNEXA] Sending data to background');
      sendToBackground(result);
      // Clear timeout to prevent re-checking
      checkTimeout = null;
    } else {
      // If not ready, keep checking every 2 seconds
      scheduleCheck();
    }
  }, 2000); // Increased to 2 seconds to reduce excessive checking
}

// Only run on Gemini pages
if (isGeminiPage) {
  // Removed MutationObserver - it was causing infinite loops
  // Now only checking on explicit user actions

  // Listen for send button clicks
  document.addEventListener('click', (e) => {
    const sendButton = e.target.closest('button.send-button, button[aria-label*="Send"]');
    if (sendButton) {
      console.log('[LUNEXA] Send button clicked, scheduling check');
      lastResponseLength = 0;
      stableCheckCount = 0;
      // Wait 3 seconds for response to start
      setTimeout(() => scheduleCheck(), 3000);
    }
  }, true);

  // Listen for Enter key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      const richTextbox = e.target.closest('[contenteditable="true"]');
      if (richTextbox && richTextbox.textContent.trim()) {
        console.log('[LUNEXA] Enter pressed, scheduling check');
        lastResponseLength = 0;
        stableCheckCount = 0;
        // Wait 3 seconds for response to start
        setTimeout(() => scheduleCheck(), 3000);
      }
    }
  }, true);

  console.log("[LUNEXA] Extension ready on Gemini (manual trigger mode)");
} else {
  console.log("[LUNEXA] Not on Gemini, idle");
}

// Listen for messages from popup (for article/selection modes)
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

  if (message.type === 'ANALYZE_ARTICLE') {

    const articleElement = document.querySelector('article');
    let articleText = "";

    if (articleElement) {
      articleText = articleElement.innerText;
    } else {
      const paragraphs = Array.from(document.querySelectorAll('p'));
      const contentParagraphs = paragraphs.filter(p => p.innerText.length > 50);
      articleText = contentParagraphs.map(p => p.innerText).join('\n\n');
    }

    if (articleText && articleText.length > 50) {
      sendToBackground({
        query: `Article Analysis: ${document.title}`,
        response: articleText.substring(0, 5000),
        mode: 'article'
      });
    }

    sendResponse({ status: 'started' });
    return false; // Response sent synchronously
  }
});

// --- Selection Tracking Logic ---
document.addEventListener('mouseup', handleSelection);
document.addEventListener('keyup', handleSelection);

function handleSelection() {
  const selection = window.getSelection().toString().trim();
  if (selection.length > 0) {
    if (chrome.runtime?.id) {
      chrome.storage.local.set({ selectedText: selection });
    }
  } else {
    // Optional: Clear selection if user clicks away? 
    // For now, let's keep the last selection to be user-friendly 
    // so they don't lose it if they accidentally click.
    // But if we want to be strict:
    // chrome.storage.local.remove('selectedText');
  }
}// --- Floating Button Injection ---
function injectFloatingButton() {
  if (document.getElementById('lunexa-floating-btn')) return;

  const btn = document.createElement('button');
  btn.id = 'lunexa-floating-btn';

  const imgUrl = chrome.runtime.getURL('lunexa.png');

  // Apply requested styles
  Object.assign(btn.style, {
    position: 'fixed',
    bottom: '110px',
    right: '380px',
    width: '40px',
    height: '40px',
    border: 'none',
    borderRadius: '50%',
    background: 'transparent',
    cursor: 'pointer',
    zIndex: '10000',
    boxShadow: 'rgba(255, 215, 0, 0.4) 0px 4px 20px',
    transition: 'transform 0.2s ease, box-shadow 0.2s',
    padding: '8px',
    transform: 'scale(1)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  });

  const img = document.createElement('img');
  img.src = imgUrl;
  img.style.width = '100%';
  img.style.height = '100%';
  img.style.objectFit = 'contain';
  img.style.pointerEvents = 'none'; // Let clicks pass to button

  btn.appendChild(img);

  // Hover effects
  btn.addEventListener('mouseenter', () => {
    btn.style.transform = 'scale(1.1)';
    btn.style.boxShadow = 'rgba(255, 215, 0, 0.6) 0px 6px 24px';
  });

  btn.addEventListener('mouseleave', () => {
    btn.style.transform = 'scale(1)';
    btn.style.boxShadow = 'rgba(255, 215, 0, 0.4) 0px 4px 20px';
  });

  // Click handler
  btn.addEventListener('click', () => {
    console.log('[LUNEXA] Floating button clicked');
    chrome.runtime.sendMessage({ type: 'OPEN_POPUP' });
  });

  document.body.appendChild(btn);
  console.log('[LUNEXA] Floating button injected');
}

// Inject immediately if body is ready, otherwise wait
if (document.body) {
  injectFloatingButton();
} else {
  document.addEventListener('DOMContentLoaded', injectFloatingButton);
}
