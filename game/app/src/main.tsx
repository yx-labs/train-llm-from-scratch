import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import { Mvp01App } from "./mvp01/Mvp01App";
import "./styles.css";

function Root() {
  const [hash, setHash] = React.useState(() => window.location.hash);

  React.useEffect(() => {
    function handleHashChange() {
      setHash(window.location.hash);
    }
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  const route = hash.replace(/^#\/?/, "").toLowerCase();
  const legacyRoute = route.startsWith("legacy") || route.startsWith("mvp0.02");

  return legacyRoute ? <App /> : <Mvp01App />;
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
