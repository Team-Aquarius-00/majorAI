/**
 * Tab Switch Detection
 * ====================
 * Detects when the user switches away from the current browser tab/window
 * Uses Page Visibility API for reliable detection
 * Terminates the session after max switches
 */

let detectionActive = false;
let tabSwitchCount = 0;
let maxSwitchesAllowed = 5;
let detectionConfig = {};
let visibilityChangeListener = null;
let blurListener = null;
let focusListener = null;
let wasHidden = false; // Track previous visibility state
let lastSwitchTime = 0; // Prevent duplicate detections
let debounceDelay = 300; // Debounce delay in ms

/**
 * Initialize tab switch detection
 * @param {Object} config - Configuration object
 *   - maxSwitches: Maximum number of switches before termination (default: 5)
 *   - initialCount: Initial count value (default: 0)
 *   - onSwitch: Callback when tab switch is detected
 *   - onTerminate: Callback when max switches exceeded
 */
function initTabSwitchDetection(config) {
  if (detectionActive) {
    console.warn("Detection already active!");
    return;
  }

  detectionConfig = config || {};
  tabSwitchCount = config.initialCount || 0;
  maxSwitchesAllowed = config.maxSwitches || 5;
  detectionActive = true;
  wasHidden = false; // Initialize visibility state
  lastSwitchTime = 0;

  console.log("Tab switch detection initialized:", {
    maxSwitches: maxSwitchesAllowed,
    initialCount: tabSwitchCount,
    timestamp: new Date().toLocaleTimeString(),
  });

  // Use Page Visibility API
  if (document.hidden !== undefined) {
    visibilityChangeListener = function () {
      handleVisibilityChange();
    };
    document.addEventListener("visibilitychange", visibilityChangeListener);
  }

  // Fallback: Use Focus/Blur events
  blurListener = function () {
    handleWindowBlur();
  };
  focusListener = function () {
    handleWindowFocus();
  };

  window.addEventListener("blur", blurListener);
  window.addEventListener("focus", focusListener);

  // Additional detection: Monitor for Alt+Tab, Ctrl+T, etc.
  document.addEventListener("keydown", handleKeyDown);
}

/**
 * Handle visibility state change (main detection)
 */
function handleVisibilityChange() {
  if (!detectionActive) return;

  const isCurrentlyHidden = document.hidden;
  const now = Date.now();

  // Only count when transitioning from visible to hidden
  if (isCurrentlyHidden && !wasHidden) {
    // Debounce: prevent multiple detections within delay
    if (now - lastSwitchTime > debounceDelay) {
      tabSwitchCount++;
      lastSwitchTime = now;

      console.log(
        `✓ Tab switch detected via Visibility API! Count: ${tabSwitchCount}/${maxSwitchesAllowed}`,
      );

      if (detectionConfig.onSwitch) {
        detectionConfig.onSwitch(tabSwitchCount);
      }

      // Add visual feedback
      document.body.style.opacity = "0.7";
      document.body.style.backgroundColor = "#ffebee";

      // Check if max switches exceeded
      if (tabSwitchCount >= maxSwitchesAllowed) {
        terminateSession(
          `You switched tabs ${tabSwitchCount} times. Maximum allowed: ${maxSwitchesAllowed}`,
        );
      }
    }
  } else if (!isCurrentlyHidden && wasHidden) {
    // User returned to this tab
    console.log("✓ User returned to tab");
    document.body.style.opacity = "1";
    document.body.style.backgroundColor = "";
  }

  wasHidden = isCurrentlyHidden;
}

/**
 * Handle window blur (Alt+Tab detection)
 */
function handleWindowBlur() {
  if (!detectionActive) return;

  const now = Date.now();

  // Detect blur event as potential tab/window switch
  if (now - lastSwitchTime > debounceDelay) {
    console.log("⚠️  Window blur detected - possible Alt+Tab");

    // Wait a bit to see if visibility change event fires
    // If it does, that will handle the count
    // If it doesn't, this ensures we still count it
    setTimeout(() => {
      if (detectionActive && !document.hidden && wasHidden === false) {
        // The page is still focused but blur was detected
        // This could be a brief focus loss
        console.log("Brief blur event detected");
      }
    }, debounceDelay);
  }
}

/**
 * Handle window focus
 */
function handleWindowFocus() {
  if (!detectionActive) return;
  console.log("✓ Window regained focus");
  // Reset visual feedback
  document.body.style.opacity = "1";
  document.body.style.backgroundColor = "";
}

/**
 * Handle keyboard shortcuts that indicate tab switching or new window
 */
function handleKeyDown(event) {
  if (!detectionActive) return;

  // Ctrl+T (new tab), Ctrl+N (new window), Ctrl+Tab (switch tab), Alt+Tab
  if (
    (event.ctrlKey && event.key === "t") || // Ctrl+T
    (event.ctrlKey && event.key === "n") || // Ctrl+N
    (event.ctrlKey && event.key === "Tab") || // Ctrl+Tab
    (event.altKey && event.key === "Tab") // Alt+Tab
  ) {
    console.log(
      "Potential tab/window switch detected via keyboard:",
      event.key,
    );
    // This might indicate cheating attempt
  }
}

/**
 * Stop tab switch detection
 */
function stopTabSwitchDetection() {
  if (!detectionActive) {
    console.warn("Detection is not active!");
    return;
  }

  detectionActive = false;
  wasHidden = false;

  // Remove event listeners
  if (visibilityChangeListener) {
    document.removeEventListener("visibilitychange", visibilityChangeListener);
  }
  if (blurListener) {
    window.removeEventListener("blur", blurListener);
  }
  if (focusListener) {
    window.removeEventListener("focus", focusListener);
  }
  document.removeEventListener("keydown", handleKeyDown);

  // Reset visual styles
  document.body.style.opacity = "1";
  document.body.style.backgroundColor = "";

  console.log("Tab switch detection stopped");
}

/**
 * Terminate the session
 */
function terminateSession(reason) {
  detectionActive = false;
  stopTabSwitchDetection();

  console.error("SESSION TERMINATED:", reason);

  if (detectionConfig.onTerminate) {
    detectionConfig.onTerminate(reason);
  }

  // Send termination event to backend (optional)
  reportTabSwitchViolation(reason);
}

/**
 * Report tab switch violation to backend
 */
function reportTabSwitchViolation(reason) {
  const reportData = {
    event: "tab_switch_violation",
    reason: reason,
    switchCount: tabSwitchCount,
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    tabTitle: document.title,
  };

  console.log("Reporting violation:", reportData);

  // Optionally send to backend API
  // fetch('/api/report-violation', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify(reportData)
  // }).catch(err => console.error('Failed to report:', err));
}

/**
 * Get current switch count
 */
function getSwitchCount() {
  return tabSwitchCount;
}

/**
 * Reset switch count
 */
function resetSwitchCount() {
  tabSwitchCount = 0;
  console.log("Switch count reset to 0");
}

/**
 * Check if detection is active
 */
function isDetectionActive() {
  return detectionActive;
}

// Export functions for use in HTML
window.initTabSwitchDetection = initTabSwitchDetection;
window.stopTabSwitchDetection = stopTabSwitchDetection;
window.getSwitchCount = getSwitchCount;
window.resetSwitchCount = resetSwitchCount;
window.isDetectionActive = isDetectionActive;
