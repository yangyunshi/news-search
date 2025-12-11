"use client";
import { useState } from "react";

interface SearchResult {
  rank: number;
  title: string;
  description: string;
  score: number;
  cluster: number;
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch("http://localhost:5000/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      setResults(data.results || []);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  const handleRebuild = async () => {
    try {
      const res = await fetch("http://127.0.0.1:5000/rebuild", { method: "POST" });
      const data = await res.json();
      alert(data.status);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ maxWidth: "700px", margin: "50px auto", fontFamily: "Arial, sans-serif" }}>
      <h1 style={{ textAlign: "center", marginBottom: "30px" }}>📰 News Search</h1>

      <form onSubmit={handleSearch} style={{ display: "flex", marginBottom: "20px" }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your search..."
          style={{
            flex: 1,
            padding: "10px",
            fontSize: "16px",
            border: "1px solid #ccc",
            borderRadius: "6px",
          }}
        />
        <button
          type="submit"
          style={{
            marginLeft: "10px",
            padding: "10px 20px",
            backgroundColor: "#0070f3",
            color: "#fff",
            fontWeight: "bold",
            border: "none",
            borderRadius: "6px",
            cursor: "pointer",
            boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
            transition: "background-color 0.3s ease",
          }}
          onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#005bb5")}
          onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "#0070f3")}
        >
          Search
        </button>
      </form>

      <button
        onClick={handleRebuild}
        style={{
          marginBottom: "20px",
          padding: "10px 20px",
          backgroundColor: "#e63946",
          color: "#fff",
          fontWeight: "bold",
          border: "none",
          borderRadius: "6px",
          cursor: "pointer",
          boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
          transition: "background-color 0.3s ease",
        }}
        onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#b71c1c")}
        onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "#e63946")}
      >
        Rebuild Pipeline
      </button>

      {loading && <p>Searching...</p>}
      <div>
        {results.length === 0 && !loading && <p>No results yet.</p>}
        {results.map((r) => (
          <div
            key={r.rank}
            style={{
              padding: "15px",
              marginBottom: "15px",
              border: "1px solid #ddd",
              borderRadius: "6px",
              backgroundColor: "#fafafa",
            }}
          >
            <h3
              style={{
                margin: "0 0 8px 0",
                fontSize: "22px",   // larger title
                fontWeight: "bold", // bold
                color: "#000",       // black
              }}
            >
              {r.rank}. {r.title}
            </h3>
            <p style={{ margin: "0 0 10px 0", color: "#555", lineHeight: "1.4" }}>
              {r.description}
            </p>
            <small style={{ color: "#888" }}>
              Cluster {r.cluster} • Score: {r.score.toFixed(2)}
            </small>
          </div>
        ))}
      </div>
    </div>
  );
}
