(function () {
  "use strict";

  const storagePrefix = "ai-dashboard-disclosure:";

  function storageKey(details) {
    const summary = details.querySelector(":scope > summary");
    return summary ? storagePrefix + summary.textContent.trim() : null;
  }

  function restore(details) {
    const key = storageKey(details);
    if (!key) return;

    try {
      const savedState = window.sessionStorage.getItem(key);
      if (savedState !== null) details.open = savedState === "open";
    } catch (_error) {
      // The disclosure still works normally when browser storage is unavailable.
    }
  }

  function restoreAll(root) {
    if (root.matches && root.matches("details.insight-card")) restore(root);
    if (root.querySelectorAll) {
      root.querySelectorAll("details.insight-card").forEach(restore);
    }
  }

  document.addEventListener(
    "toggle",
    function (event) {
      const details = event.target;
      if (!details.matches || !details.matches("details.insight-card")) return;

      const key = storageKey(details);
      if (!key) return;
      try {
        window.sessionStorage.setItem(key, details.open ? "open" : "closed");
      } catch (_error) {
        // Ignore storage restrictions without affecting the control itself.
      }
    },
    true
  );

  const observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (mutation) {
      mutation.addedNodes.forEach(function (node) {
        if (node.nodeType === Node.ELEMENT_NODE) restoreAll(node);
      });
    });
  });

  function start() {
    restoreAll(document);
    observer.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
