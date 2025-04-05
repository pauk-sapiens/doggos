// Get references to elements
const dropArea = document.getElementById("dropArea")
const fileInput = document.getElementById("file-upload")
const imagePreview = document.getElementById("image-preview")
const dropInstructions = document.getElementById("dropInstructions")
const previewContainer = document.getElementById("previewContainer")
const removeImageBtn = document.getElementById("removeImage")
const form = document.getElementById("uploadForm")

// Handle drag and drop functionality
if (dropArea && fileInput) {
  // Highlight drop area on dragenter / dragover
  ;["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(
      eventName,
      (e) => {
        e.preventDefault()
        e.stopPropagation()
        if (!previewContainer.classList.contains("active")) {
          dropArea.classList.add("highlight")
        }
      },
      false,
    )
  })

  // Remove highlight on dragleave / drop
  ;["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(
      eventName,
      (e) => {
        e.preventDefault()
        e.stopPropagation()
        dropArea.classList.remove("highlight")
      },
      false,
    )
  })

  // Handle dropped files
  dropArea.addEventListener("drop", (e) => {
    const dt = e.dataTransfer
    const files = dt.files

    if (files.length > 0) {
      fileInput.files = files // Assign dropped files to the input element
      handleFileSelection(files[0]) // Process the first file
    }
  })
}

// Handle file input change
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0]
  if (file) {
    handleFileSelection(file)
  }
})

// Function to handle file selection (from both drag & drop and file input)
function handleFileSelection(file) {
  if (file) {
    const reader = new FileReader()

    reader.onload = (e) => {
      // Set the image source
      imagePreview.src = e.target.result

      // Hide drop instructions and show preview
      dropInstructions.style.display = "none"
      previewContainer.classList.add("active")
    }

    reader.readAsDataURL(file)
  }
}

// Remove image button functionality
if (removeImageBtn) {
  removeImageBtn.addEventListener("click", (e) => {
    e.preventDefault()
    e.stopPropagation() // Prevent triggering the label click

    // Clear the file input
    fileInput.value = ""

    // Hide preview and show drop instructions
    previewContainer.classList.remove("active")
    dropInstructions.style.display = "flex"
  })
}

// Reset the upload area on form submit
form.addEventListener("submit", () => {
  // The form will be submitted and page will refresh,
  // but if you're using AJAX, you might want to reset the upload area
  // previewContainer.classList.remove("active");
  // dropInstructions.style.display = "flex";
})

