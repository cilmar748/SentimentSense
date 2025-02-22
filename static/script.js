// script.js

// DOM Elements
const form = document.getElementById("sentimentForm");
const textInput = document.getElementById("textInput");
const charCount = document.getElementById("charCount");
const analyzeButton = document.getElementById("analyzeButton");
const loadingSpinner = document.getElementById("loadingSpinner");
const toast = document.getElementById("toast");

// Character Count Update
textInput.addEventListener("input", () => {
    const textLength = textInput.value.length;
    charCount.textContent = `${textLength}/500 characters`;
    if (textLength > 500) {
        charCount.style.color = "#ff3b2f"; // Red for exceeding limit
    } else {
        charCount.style.color = "#e0e0e0"; // Default color
    }
});

// Form Submission Handling
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Disable button and show loading spinner
    analyzeButton.disabled = true;
    loadingSpinner.style.display = "block";

    try {
        const formData = new FormData(form);
        console.log("Form Data:", formData); // Debugging

        const response = await fetch("/analyze", {
            method: "POST",
            body: formData,
        });

        console.log("Response:", response); // Debugging

        if (!response.ok) {
            throw new Error("Failed to analyze sentiment");
        }

        const data = await response.json();
        console.log("Data:", data); // Debugging

        // Clear textarea after submission
        textInput.value = "";
        charCount.textContent = "0/500 characters";

        // Display the sentiment result
        const resultDiv = document.createElement("div");
        resultDiv.className = "result";
        resultDiv.innerHTML = `
            <h2>Result</h2>
            <p>Sentiment: <span class="sentiment ${data.sentiment.toLowerCase()}">${data.sentiment}</span></p>
        `;

        // Remove existing result if any
        const existingResult = document.querySelector(".result");
        if (existingResult) {
            existingResult.remove();
        }

        // Append new result
        form.insertAdjacentElement("afterend", resultDiv);

        // Show success toast
        showToast("Sentiment analyzed successfully!", "success");
    } catch (error) {
        console.error("Error:", error); // Debugging
        showToast("An error occurred. Please try again.", "error");
    } finally {
        // Re-enable button and hide spinner
        analyzeButton.disabled = false;
        loadingSpinner.style.display = "none";
    }
});

// Toast Notification Function
function showToast(message, type) {
    toast.textContent = message;
    toast.className = "toast show"; // Reset and show toast

    // Add type-specific styling
    if (type === "success") {
        toast.style.backgroundColor = "#4caf50";
    } else if (type === "error") {
        toast.style.backgroundColor = "#f44336";
    }

    // Hide toast after 3 seconds
    setTimeout(() => {
        toast.className = toast.className.replace("show", "");
    }, 3000);
}