// Frontend Client Logic for Customer Support Ticket Classifier
document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const ticketForm = document.getElementById("ticketForm");
  const ticketDescription = document.getElementById("ticketDescription");
  const customerName = document.getElementById("customerName");
  const ticketPriority = document.getElementById("ticketPriority");
  const charCount = document.getElementById("charCount");
  const sampleChips = document.getElementById("sampleChips");
  
  const btnPredict = document.getElementById("btnPredict");
  const btnSpinner = document.getElementById("btnSpinner");
  const btnClear = document.getElementById("btnClear");
  
  const emptyState = document.getElementById("emptyState");
  const outputContent = document.getElementById("outputContent");
  
  const categoryBanner = document.getElementById("categoryBanner");
  const categoryIcon = document.getElementById("categoryIcon");
  const predictedCategory = document.getElementById("predictedCategory");
  const slaBadge = document.getElementById("slaBadge");
  const confidenceValue = document.getElementById("confidenceValue");
  const confidenceFill = document.getElementById("confidenceFill");
  
  const actionText = document.getElementById("actionText");
  const responseText = document.getElementById("responseText");
  const btnCopyResponse = document.getElementById("btnCopyResponse");
  const copyLabel = document.getElementById("copyLabel");
  const probBarsContainer = document.getElementById("probBarsContainer");

  // Threshold and Fallback Elements
  const thresholdSlider = document.getElementById("thresholdSlider");
  const thresholdDisplay = document.getElementById("thresholdDisplay");
  const fallbackAlert = document.getElementById("fallbackAlert");
  const fallbackConfVal = document.getElementById("fallbackConfVal");
  const fallbackThreshVal = document.getElementById("fallbackThreshVal");
  const heroHeading = document.getElementById("heroHeading");
  const triageBadge = document.getElementById("triageBadge");
  const categorySubtext = document.getElementById("categorySubtext");
  const confStatusTag = document.getElementById("confStatusTag");

  // Inline Result Elements (Below Predict Button)
  const inlineResult = document.getElementById("inlineResult");
  const inlineTag = document.getElementById("inlineTag");
  const inlinePredLabel = document.getElementById("inlinePredLabel");
  const inlinePredictedCategory = document.getElementById("inlinePredictedCategory");
  const inlineConfidence = document.getElementById("inlineConfidence");

  // Category Icon & Styling Map
  const CATEGORY_META = {
    "Login Issue": { icon: "🔑", class: "cat-login-issue" },
    "Application Error": { icon: "⚠️", class: "cat-application-error" },
    "Report": { icon: "📊", class: "cat-report" },
    "Account Update": { icon: "👤", class: "cat-account-update" },
    "Performance": { icon: "⚡", class: "cat-performance" },
    "Payment Issue": { icon: "💳", class: "cat-payment-issue" },
    "Access Issue": { icon: "🛡️", class: "cat-access-issue" },
    "Data Issue": { icon: "🗄️", class: "cat-data-issue" },
  };

  // Sample Presets for Quick Testing (Including Vague Ticket for Fallback Verification)
  const SAMPLES = [
    { label: "Vague (Not Working)", text: "application is not working properly", priority: "Medium" },
    { label: "Login Reset", text: "I forgot my password and cannot sign into my account.", priority: "High" },
    { label: "App Crash", text: "The application gives a 500 error and crashes when clicking save.", priority: "Critical" },
    { label: "Sales Report", text: "Please help me generate the monthly sales and revenue report.", priority: "Low" },
    { label: "Email Change", text: "I need to update my registered email and phone number.", priority: "Medium" },
    { label: "Slow Loading", text: "The application performance has become poor and pages load slowly.", priority: "High" },
    { label: "Payment Failed", text: "Payment was deducted from my account but order is still pending.", priority: "Critical" },
    { label: "Access Denied", text: "Access denied when opening the admin settings dashboard.", priority: "High" },
    { label: "Data Missing", text: "Customer records entered yesterday are no longer visible in system.", priority: "Medium" },
  ];

  // Initialize Threshold Slider Listener
  if (thresholdSlider && thresholdDisplay) {
    thresholdSlider.addEventListener("input", () => {
      thresholdDisplay.textContent = `${thresholdSlider.value}%`;
      // If we have text in the box and results are showing, re-run prediction seamlessly
      if (ticketDescription.value.trim() && !outputContent.classList.contains("hidden")) {
        handlePredict();
      }
    });
  }

  // Initialize Sample Chips
  function renderSampleChips() {
    sampleChips.innerHTML = "";
    SAMPLES.forEach(sample => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "sample-chip";
      chip.textContent = sample.label;
      chip.addEventListener("click", () => {
        ticketDescription.value = sample.text;
        if (ticketPriority) ticketPriority.value = sample.priority;
        updateCharCount();
        handlePredict();
      });
      sampleChips.appendChild(chip);
    });
  }

  // Update Character Count
  function updateCharCount() {
    const len = ticketDescription.value.length;
    charCount.textContent = `${len} character${len === 1 ? '' : 's'}`;
  }

  ticketDescription.addEventListener("input", updateCharCount);

  // Clear Form
  btnClear.addEventListener("click", () => {
    ticketDescription.value = "";
    customerName.value = "";
    if (ticketPriority) ticketPriority.value = "Medium";
    updateCharCount();
    emptyState.classList.remove("hidden");
    outputContent.classList.add("hidden");
    if (inlineResult) inlineResult.classList.add("hidden");
  });

  // Handle Ticket Submission
  async function handlePredict(e) {
    if (e) e.preventDefault();
    const text = ticketDescription.value.trim();
    if (!text) {
      ticketDescription.focus();
      return;
    }

    // Set Loading State
    btnPredict.disabled = true;
    btnSpinner.style.display = "inline-block";

    const thresholdVal = thresholdSlider ? parseFloat(thresholdSlider.value) : 40.0;

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          description: text,
          customer_name: customerName.value.trim() || null,
          priority: ticketPriority ? ticketPriority.value : "Medium",
          confidence_threshold: thresholdVal,
        })
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Server inference error");
      }

      const data = await response.json();
      displayResults(data);
    } catch (err) {
      alert(`Prediction Failed: ${err.message}`);
    } finally {
      btnPredict.disabled = false;
      btnSpinner.style.display = "none";
    }
  }

  // Display Results
  function displayResults(data) {
    emptyState.classList.add("hidden");
    outputContent.classList.remove("hidden");

    // Category Meta & Styling
    const meta = CATEGORY_META[data.predicted_category] || { icon: "📌", class: "" };
    const isFallback = Boolean(data.is_low_confidence);

    if (isFallback) {
      // 1. Show Low Confidence Warning Banner
      if (fallbackAlert) {
        fallbackAlert.classList.remove("hidden");
        if (fallbackConfVal) fallbackConfVal.textContent = `${data.confidence.toFixed(1)}%`;
        if (fallbackThreshVal) fallbackThreshVal.textContent = `${data.confidence_threshold.toFixed(1)}%`;
      }

      // 2. Adjust Hero Routing
      if (heroHeading) heroHeading.textContent = "Assigned Routing:";
      if (triageBadge) {
        triageBadge.className = "triage-badge triage-fallback";
        triageBadge.textContent = "⚠️ Manual Triage Required";
      }

      categoryIcon.textContent = "⚠️";
      predictedCategory.textContent = data.routing_category || "Needs Clarification / Manual Review";
      predictedCategory.className = "category-name cat-fallback";

      if (categorySubtext) {
        categorySubtext.classList.remove("hidden");
        categorySubtext.innerHTML = `Model Suggestion: <strong>${meta.icon} ${data.predicted_category}</strong> (Uncertain: ${data.confidence.toFixed(1)}% &lt; ${data.confidence_threshold.toFixed(1)}% threshold)`;
      }

      // 3. Confidence Status Tag & Bar
      if (confStatusTag) {
        confStatusTag.className = "conf-status-tag conf-status-low";
        confStatusTag.textContent = "Low Confidence";
      }
      confidenceFill.className = "progress-fill fill-warning";

      // 4. Update Inline Instant Result
      if (inlineResult && inlinePredictedCategory && inlineConfidence) {
        if (inlineTag) inlineTag.textContent = "⚠️ Fallback Active";
        if (inlinePredLabel) inlinePredLabel.textContent = "Assigned Routing:";
        inlinePredictedCategory.innerHTML = `<span style="font-size: 1.15rem;">⚠️</span> <span class="cat-fallback">Needs Review</span> <span style="font-size: 0.78rem; color: #9ca3af;">(Top ML: ${data.predicted_category})</span>`;
        inlineConfidence.textContent = `Confidence: ${data.confidence.toFixed(1)}% (< ${data.confidence_threshold.toFixed(1)}%)`;
        inlineResult.classList.remove("hidden");
      }
    } else {
      // Standard Confident Prediction
      if (fallbackAlert) fallbackAlert.classList.add("hidden");

      if (heroHeading) heroHeading.textContent = "Predicted Category:";
      if (triageBadge) {
        triageBadge.className = "triage-badge triage-automated";
        triageBadge.textContent = "Automated Dispatch";
      }

      categoryIcon.textContent = meta.icon;
      predictedCategory.textContent = data.predicted_category;
      predictedCategory.className = `category-name ${meta.class}`;

      if (categorySubtext) categorySubtext.classList.add("hidden");

      if (confStatusTag) {
        confStatusTag.className = "conf-status-tag conf-status-confident";
        confStatusTag.textContent = "Confident Dispatch";
      }
      confidenceFill.className = "progress-fill";

      // Update Inline Instant Result
      if (inlineResult && inlinePredictedCategory && inlineConfidence) {
        if (inlineTag) inlineTag.textContent = "Instant Prediction";
        if (inlinePredLabel) inlinePredLabel.textContent = "Predicted Category:";
        inlinePredictedCategory.innerHTML = `<span style="font-size: 1.3rem;">${meta.icon}</span> <span class="${meta.class}">${data.predicted_category}</span>`;
        inlineConfidence.textContent = `Confidence: ${data.confidence.toFixed(1)}%`;
        inlineResult.classList.remove("hidden");
      }
    }

    // Trigger subtle pulse animation on inline card
    if (inlineResult) {
      inlineResult.style.animation = "none";
      inlineResult.offsetHeight; // Trigger reflow
      inlineResult.style.animation = "resultGlow 0.4s ease-out";
    }

    // SLA & Confidence Values
    slaBadge.textContent = `SLA: ${data.estimated_sla.split('(')[0].trim()}`;
    confidenceValue.textContent = `${data.confidence.toFixed(1)}%`;
    confidenceFill.style.width = `${Math.min(data.confidence, 100)}%`;

    // Action & Automated Response
    actionText.textContent = data.recommended_action;
    responseText.textContent = data.suggested_response;

    // Probability Bars
    probBarsContainer.innerHTML = "";
    Object.entries(data.all_probabilities).forEach(([cat, prob]) => {
      const row = document.createElement("div");
      row.className = "prob-bar-row";

      const nameSpan = document.createElement("span");
      nameSpan.className = "prob-bar-name";
      nameSpan.textContent = cat;

      const trackDiv = document.createElement("div");
      trackDiv.className = "prob-bar-track";

      const fillDiv = document.createElement("div");
      fillDiv.className = "prob-bar-fill";
      if (cat === data.predicted_category) {
        fillDiv.style.background = isFallback
          ? "linear-gradient(90deg, #f59e0b, #ef4444)"
          : "linear-gradient(90deg, #6366f1, #38bdf8)";
      } else {
        fillDiv.style.background = "rgba(255, 255, 255, 0.25)";
      }
      fillDiv.style.width = `${prob}%`;
      trackDiv.appendChild(fillDiv);

      const valSpan = document.createElement("span");
      valSpan.className = "prob-bar-val";
      valSpan.textContent = `${prob.toFixed(1)}%`;

      row.appendChild(nameSpan);
      row.appendChild(trackDiv);
      row.appendChild(valSpan);
      probBarsContainer.appendChild(row);
    });
  }

  // Copy Response to Clipboard
  btnCopyResponse.addEventListener("click", () => {
    const textToCopy = responseText.textContent;
    if (!textToCopy) return;

    navigator.clipboard.writeText(textToCopy).then(() => {
      copyLabel.textContent = "Copied!";
      setTimeout(() => {
        copyLabel.textContent = "Copy";
      }, 2000);
    }).catch(() => {
      // Fallback
      copyLabel.textContent = "Copied!";
      setTimeout(() => { copyLabel.textContent = "Copy"; }, 2000);
    });
  });

  ticketForm.addEventListener("submit", handlePredict);
  btnPredict.addEventListener("click", (e) => {
    // If inside form, let submit handle it, or call handlePredict directly
    handlePredict(e);
  });

  // Initialize
  renderSampleChips();
  updateCharCount();
});
