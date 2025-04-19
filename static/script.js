document.getElementById("predictBtn").addEventListener("click", function () {
    fetch("/predict")
        .then((response) => response.json())
        .then((data) => {
            let output = "<h3>Forecast Results</h3><table border='1'><tr><th>Category</th><th>Predicted Sales</th><th>Reorder Level</th><th>Restock Quantity</th></tr>";
            data.forEach((row) => {
                output += `<tr><td>${row.ProductCategory}</td><td>${row["Predicted Sales"]}</td><td>${row["Reorder Level"]}</td><td>${row["Restock Quantity"]}</td></tr>`;
            });
            output += "</table>";
            document.getElementById("results").innerHTML = output;
        })
        .catch((error) => console.error("Error:", error));
});
