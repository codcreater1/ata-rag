import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";

import Chat from "./Chat";
import Dashboard from "./Dashboard";
import "./styles.css";

function App() {
  const [view, setView] = useState("chat");

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">🐻</span>
          <div>
            <strong>ATA Assistant</strong>
            <small>Akademia Techniczno-Artystyczna</small>
          </div>
        </div>
        <nav className="tabs">
          <button
            className={view === "chat" ? "tab on" : "tab"}
            onClick={() => setView("chat")}
          >
            Chat
          </button>
          <button
            className={view === "dashboard" ? "tab on" : "tab"}
            onClick={() => setView("dashboard")}
          >
            Dashboard
          </button>
        </nav>
      </header>

      <main className="main">
        {view === "chat" ? <Chat /> : <Dashboard />}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
