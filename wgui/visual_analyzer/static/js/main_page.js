function toggleInput(inputType) {
	if (inputType === 'url') {
			document.getElementById('url-input').style.display = 'block';
			document.getElementById('file-input').style.display = 'none';
	} else if (inputType === 'file') {
			document.getElementById('url-input').style.display = 'none';
			document.getElementById('file-input').style.display = 'block';
	}
}

function clearForm() {
	document.querySelector('form').reset();
	document.getElementById('url-input').style.display = 'block';
	document.getElementById('file-input').style.display = 'none';
	document.getElementById('url').value = '';
	document.getElementById('file').value = '';
	window.location.href = window.location.pathname;
}

function triggerSpinner() {
	document.title = "Searching...";
	document.getElementById("loadingSpinner").style.display = "flex";
}

function hideLoadingSpinner() {
	document.getElementById("loadingSpinner").style.display = "none";
}