function searchMovie() {
  const movie = document.getElementById("movieInput").value.trim();
  if (!movie) return;

  fetch("http://127.0.0.1:5000/recommend", { 
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ movie_name: movie }),
  })
    .then((response) => response.json())
    .then((data) => {
      const resultsList = document.getElementById("results");
      resultsList.innerHTML = "";
      if (data.movies && data.movies.length > 0) {
        data.movies.forEach((m) => {
          const li = document.createElement("li");
          li.textContent = m;
          resultsList.appendChild(li);
        });
      } else {
        resultsList.innerHTML = "<li>No similar movies found.</li>";
      }
    })
    .catch(() => {
      document.getElementById("results").innerHTML =
        "<li>Error fetching recommendations.</li>";
    });
}