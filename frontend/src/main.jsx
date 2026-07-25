import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";

import Chat from "./Chat";
import Dashboard from "./Dashboard";
import "./styles.css";

// The site itself is published in these languages; "auto" mirrors whatever the
// visitor typed, which is the sensible default.
const LANGUAGES = [
  { code: "auto", label: "Auto" },
  { code: "en", label: "EN" },
  { code: "pl", label: "PL" },
  { code: "uk", label: "UK" },
];

function App() {
  const [view, setView] = useState("chat");
  const [language, setLanguage] = useState("auto");

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

        <div className="topRight">
          {view === "chat" && (
            <div className="langPicker" role="group" aria-label="Answer language">
              {LANGUAGES.map((l) => (
                <button
                  key={l.code}
                  className={language === l.code ? "lang on" : "lang"}
                  onClick={() => setLanguage(l.code)}
                  aria-pressed={language === l.code}
                  title={
                    l.code === "auto"
                      ? "Answer in the language of the question"
                      : `Answer in ${l.label}`
                  }
                >
                  {l.label}
                </button>
              ))}
            </div>
          )}

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
        </div>
      </header>

      <main className="main">
        {view === "chat" ? <Chat language={language} /> : <Dashboard />}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
