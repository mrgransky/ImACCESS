function toggleInput(inputType) {
	if (inputType === 'url') {
			document.getElementById('url-input').style.display = 'block';
			document.getElementById('file-input').style.display = 'none';
	} else if (inputType === 'file') {
			document.getElementById('url-input').style.display = 'none';
			document.getElementById('file-input').style.display = 'block';
	}
}

// Function to clear the form and reset inputs
function clearForm() {
	// Reset the form
	document.querySelector('form').reset();

	// Ensure that URL input is visible and file input is hidden
	document.getElementById('url-input').style.display = 'block';
	document.getElementById('file-input').style.display = 'none';

	// Reload the page to clear any context or displayed data
	window.location.href = window.location.pathname;
}

// Function to handle image preview (URL or File) when "Go" is clicked
function previewImage(event) {
	event.preventDefault();  // Prevent form submission

	// Get the selected input type (URL or file)
	const inputType = document.querySelector('input[name="input_type"]:checked').value;

	// Get the image preview container
	const imagePreview = document.getElementById('image-preview');

	if (inputType === 'url') {
		// Display image from URL
		const imageUrl = document.getElementById('url').value;
		if (imageUrl) {
			imagePreview.src = imageUrl;
			imagePreview.style.display = 'block';
		}
	} else if (inputType === 'file') {
		// Display uploaded image
		const imageFile = document.getElementById('file').files[0];
		if (imageFile) {
			const reader = new FileReader();
			reader.onload = function(e) {
				imagePreview.src = e.target.result;
				imagePreview.style.display = 'block';
			};
			reader.readAsDataURL(imageFile);
		}
	}
}