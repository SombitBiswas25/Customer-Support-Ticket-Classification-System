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

  // Sample Presets for Quick Testing
  const SAMPLES = [
    { label: "Login Reset", text: "I forgot my password and cannot sign into my account.", priority: "High" },
    { label: "App Crash", text: "The application gives a 500 error and crashes when clicking save.", priority: "Critical" },
    { label: "Sales Report", text: "Please help me generate the monthly sales and revenue report.", priority: "Low" },
    { label: "Email Change", text: "I need to update my registered email and phone number.", priority: "Medium" },
    { label: "Slow Loading", text: "The application performance has become poor and pages load slowly.", priority: "High" },
    { label: "Payment Failed", text: "Payment was deducted from my account but order is still pending.", priority: "Critical" },
    { label: "Access Denied", text: "Access denied when opening the admin settings dashboard.", priority: "High" },
    { label: "Data Missing", text: "Customer records entered yesterday are no longer visible in system.", priority: "Medium" },
  ];

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
        ticketPriority.value = sample.priority;
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
    ticketPriority.value = "Medium";
    updateCharCount();
    emptyState.classList.remove("hidden");
    outputContent.classList.add("hidden");
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

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          description: text,
          customer_name: customerName.value.trim() || null,
          priority: ticketPriority.value
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
    categoryIcon.textContent = meta.icon;
    predictedCategory.textContent = data.predicted_category;
    
    // Reset classes and apply category class
    predictedCategory.className = `category-name ${meta.class}`;

    // SLA & Confidence
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
        fillDiv.style.background = "linear-gradient(90deg, #6366f1, #38bdf8)";
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

  // Initialize
  renderSampleChips();
  updateCharCount();
});
