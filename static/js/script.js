const fileInput = document.getElementById("dataset");

const uploadBox = document.querySelector(".upload-box");

fileInput.addEventListener("change", function () {

    if (this.files.length > 0) {

        uploadBox.querySelector("h3").textContent =
            this.files[0].name;

    }

});