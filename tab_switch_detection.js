/**
 * Tab Switch Detection Module
 * ============================
 * Detects when a user switches tabs or windows during a session (e.g., an exam or interview).
 * Optionally reports each switch to a backend endpoint.
 * Auto-terminates the session after a configurable maximum number of switches.
 *
 * Usage:
 *   Include this script in your HTML page, then call initTabSwitchDetection().
 *
 * Example:
 *   <script src="tab_switch_detection.js"></script>
 *   <script>
 *     initTabSwitchDetection({
 *       maxSwitches: 5,
 *       onSwitch: (count) => {
 *         document.getElementById("switch-count").textContent = count;
 *         alert("Warning: Tab switch #" + count + " detected!");
 *       },
 *       onTerminate: () => {
 *         alert("Session terminated due to too many tab switches.");
 *         window.location.href = "/session-ended/";
 *       },
 *       recordUrl: "/api/record-tab-switch/",   // optional — POST to backend on each switch
 *       csrfToken: "YOUR_CSRF_TOKEN"             // optional — needed if using Django CSRF
 *     });
 *   </script>
 *
 * Features:
 *   - Counts tab/window visibility changes (only visible→hidden transitions)
 *   - Prevents right-click context menu
 *   - Requests fullscreen and re-requests if user exits
 *   - Warns user before closing/refreshing the page
 *   - All listeners are cleanly removed when session ends (call stopTabSwitchDetection())
 */

(function () {
  let tabSwitchCount = 0;
  let lastVisibilityState = document.visibilityState;
  let _config = {};
  let _tabSwitchHandler = null;
  let _fullscreenHandler = null;
  let _blurHandler = null;
  let _contextMenuHandler = null;
  let _beforeUnloadHandler = null;

  /**
   * Initialize tab switch detection.
   *
   * @param {Object} config
   * @param {number}   config.maxSwitches    - Max allowed tab switches before termination. Default: 5
   * @param {number}   config.initialCount   - Starting count (e.g. loaded from server). Default: 0
   * @param {Function} config.onSwitch       - Called with (count) on each tab switch
   * @param {Function} config.onTerminate    - Called when maxSwitches is exceeded
   * @param {string}   [config.recordUrl]    - Optional backend URL to POST switch events to
   * @param {string}   [config.csrfToken]    - Optional CSRF token for Django/Rails backends
   * @param {boolean}  [config.fullscreen]   - Whether to request and enforce fullscreen. Default: true
   * @param {boolean}  [config.warnOnClose]  - Whether to warn before page close/refresh. Default: true
   */
  window.initTabSwitchDetection = function (config) {
    _config = Object.assign(
      {
        maxSwitches: 5,
        initialCount: 0,
        onSwitch: null,
        onTerminate: null,
        recordUrl: null,
        csrfToken: null,
        fullscreen: true,
        warnOnClose: true,
      },
      config,
    );

    tabSwitchCount = _config.initialCount;

    // --- Fullscreen ---
    if (_config.fullscreen) {
      if (document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen().catch(() => {});
      }

      _fullscreenHandler = function () {
        if (!document.fullscreenElement) {
          alert("You cannot exit fullscreen during this session!");
          document.documentElement.requestFullscreen().catch(() => {});
        }
      };
      document.addEventListener("fullscreenchange", _fullscreenHandler);
    }

    // --- Prevent right-click ---
    _contextMenuHandler = e => e.preventDefault();
    document.addEventListener("contextmenu", _contextMenuHandler);

    // --- Blur alert (window loses focus) ---
    _blurHandler = function () {
      alert("You cannot switch tabs or windows during this session!");
      window.focus();
    };
    window.addEventListener("blur", _blurHandler);

    // --- Tab switch via Page Visibility API (most reliable method) ---
    _tabSwitchHandler = function () {
      // Only count visible → hidden transitions (not hidden → visible)
      if (
        lastVisibilityState === "visible" &&
        document.visibilityState === "hidden"
      ) {
        tabSwitchCount++;

        // Notify backend (optional)
        if (_config.recordUrl) {
          const headers = { "Content-Type": "application/json" };
          if (_config.csrfToken) headers["X-CSRFToken"] = _config.csrfToken;

          fetch(_config.recordUrl, { method: "POST", headers })
            .then(res => res.json())
            .then(data => {
              if (data.status === "terminated") {
                _terminate(data.message);
              }
            })
            .catch(err => console.warn("Tab switch report failed:", err));
        }

        // Check termination threshold
        if (tabSwitchCount > _config.maxSwitches) {
          _terminate("Session terminated: too many tab switches.");
          return;
        }

        // Fire user callback
        if (typeof _config.onSwitch === "function") {
          _config.onSwitch(tabSwitchCount);
        }
      }
      lastVisibilityState = document.visibilityState;
    };
    document.addEventListener("visibilitychange", _tabSwitchHandler);

    // --- Warn before close/refresh ---
    if (_config.warnOnClose) {
      _beforeUnloadHandler = function () {
        return "You are in an active session. Are you sure you want to leave?";
      };
      window.onbeforeunload = _beforeUnloadHandler;
    }
  };

  /**
   * Stop all tab switch detection listeners and clean up.
   * Call this when the user legitimately ends the session (e.g., on form submit).
   */
  window.stopTabSwitchDetection = function () {
    window.onbeforeunload = null;
    if (_tabSwitchHandler)
      document.removeEventListener("visibilitychange", _tabSwitchHandler);
    if (_fullscreenHandler)
      document.removeEventListener("fullscreenchange", _fullscreenHandler);
    if (_blurHandler) window.removeEventListener("blur", _blurHandler);
    if (_contextMenuHandler)
      document.removeEventListener("contextmenu", _contextMenuHandler);
  };

  /**
   * Get the current tab switch count.
   * @returns {number}
   */
  window.getTabSwitchCount = function () {
    return tabSwitchCount;
  };

  function _terminate(message) {
    stopTabSwitchDetection();
    if (typeof _config.onTerminate === "function") {
      _config.onTerminate(message);
    }
  }
})();
