function setupDropZone(dropZoneId, inputId, previewId) {
    const dropZone = document.getElementById(dropZoneId);
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
  
    // Click to open file
    dropZone.addEventListener("click", () => input.click());
  
    // Drag over effect
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  
    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragover");
    });
  
    // Drop file
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) {
        input.files = e.dataTransfer.files;
        showPreview(preview, file);
      }
    });
  
    // Change from input
    input.addEventListener("change", () => {
      const file = input.files[0];
      if (file && file.type.startsWith("image/")) {
        showPreview(preview, file);
      }
    });
  
    function showPreview(previewElement, file) {
      previewElement.src = URL.createObjectURL(file);
      previewElement.style.display = "block";
    }
  }
  
  // Setup both drop zones
  setupDropZone("drop-left", "left_image", "left_preview");
  setupDropZone("drop-right", "right_image", "right_preview");