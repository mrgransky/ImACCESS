function triggerSpinner() {
	document.title = "Searching...";
	document.getElementById("loadingSpinner").style.display = "flex";
}

function hideLoadingSpinner() {
	document.getElementById("loadingSpinner").style.display = "none";
}