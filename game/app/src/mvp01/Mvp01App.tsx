import { BookOpenText, RotateCcw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { graphText, type GraphLanguage } from "../gameplayGraph/i18n";
import { GraphWorkbench, type ComponentAvailabilityPayload, type ComponentFlowLevelSpec } from "../gameplayGraph/ui/GraphWorkbench";
import { mvp01ComponentFlowSpecs, mvp01GraphLevels } from "./mvp01GraphLevels";

const availableStorageKey = "llm-complete:mvp0.1:available-components";
const legacyPackedStorageKey = "llm-complete:mvp0.1:packed-components";
const graphLanguageStorageKey = "llm-complete.graph-language";

type AvailableComponentRecord = {
  id: string;
  sourceLevelId: string;
  title: string;
  version: number;
  status: "available";
  exportedModuleId: string;
  implementationGraph: ComponentAvailabilityPayload["graph"];
  certification: {
    visible: ComponentAvailabilityPayload["visible"];
    hidden: ComponentAvailabilityPayload["hidden"];
  };
  createdAt: number;
  updatedAt: number;
};

export function Mvp01App() {
  const [language, setLanguage] = useState<GraphLanguage>(() => readGraphLanguage());
  const [availableComponents, setAvailableComponents] = useState<AvailableComponentRecord[]>(() => readAvailableComponents());
  const availableComponentIds = useMemo(() => availableComponents.map((component) => component.id), [availableComponents]);
  const t = (text: string) => graphText(language, text);
  const componentSpecs = useMemo(
    () =>
      Object.fromEntries(
        mvp01ComponentFlowSpecs.map((component): [string, ComponentFlowLevelSpec] => [
          component.levelId,
          {
            componentId: component.componentId,
            title: component.title,
            version: component.version,
            exportModuleId: component.exportModuleId,
            requires: component.requires,
            unlocks: component.unlocks,
            shelf: component.shelf
          }
        ])
      ),
    []
  );

  useEffect(() => {
    window.localStorage.setItem(availableStorageKey, JSON.stringify(availableComponents));
  }, [availableComponents]);

  useEffect(() => {
    window.localStorage.setItem(graphLanguageStorageKey, language);
  }, [language]);

  function markComponentAvailable(payload: ComponentAvailabilityPayload) {
    const now = Date.now();
    setAvailableComponents((current) => {
      const existing = current.find((component) => component.id === payload.component.componentId);
      const next: AvailableComponentRecord = {
        id: payload.component.componentId,
        sourceLevelId: payload.level.id,
        title: payload.component.title,
        version: payload.component.version,
        status: "available",
        exportedModuleId: payload.component.exportModuleId,
        implementationGraph: payload.graph,
        certification: {
          visible: payload.visible,
          hidden: payload.hidden
        },
        createdAt: existing?.createdAt ?? now,
        updatedAt: now
      };
      return existing ? current.map((component) => (component.id === next.id ? next : component)) : [...current, next];
    });
  }

  function resetRoute() {
    window.localStorage.removeItem(availableStorageKey);
    window.localStorage.removeItem(legacyPackedStorageKey);
    window.location.reload();
  }

  return (
    <main className="mvp01GraphShell">
      <header className="mvp01GraphTopbar">
        <div>
          <p className="eyebrow">{t("MVP0.1 Component Builder")}</p>
          <h1>Chapter 0-9 Component Map</h1>
          <small>{t("Build key LLM components deeply; design-ready roadmap nodes stay visible but cannot be entered until implemented.")}</small>
        </div>
        <nav className="mvp01GraphActions" aria-label="MVP0.1 navigation">
          <div className="modeSwitch mvp01LanguageSwitch" aria-label={t("Language")}>
            <button type="button" className={language === "en" ? "modeButton active" : "modeButton"} aria-pressed={language === "en"} onClick={() => setLanguage("en")}>
              EN
            </button>
            <button type="button" className={language === "zh" ? "modeButton active" : "modeButton"} aria-pressed={language === "zh"} onClick={() => setLanguage("zh")}>
              中文
            </button>
          </div>
          <a className="ghostButton mvp01LegacyLink" href="#/legacy" title={t("Open legacy prototype")}>
            <BookOpenText size={15} />
            {t("Legacy")}
          </a>
          <button className="ghostButton" type="button" title={t("Reset MVP0.1 component builder")} onClick={resetRoute}>
            <RotateCcw size={15} />
            {t("Reset Route")}
          </button>
        </nav>
      </header>
      <GraphWorkbench
        language={language}
        levels={mvp01GraphLevels}
        componentFlow={{
          specs: componentSpecs,
          availableComponentIds,
          onComponentAvailable: markComponentAvailable
        }}
      />
    </main>
  );
}

function readGraphLanguage(): GraphLanguage {
  try {
    const saved = window.localStorage.getItem(graphLanguageStorageKey);
    if (saved === "en" || saved === "zh") return saved;
    return "zh";
  } catch {
    return "zh";
  }
}

function readAvailableComponents(): AvailableComponentRecord[] {
  try {
    const raw = window.localStorage.getItem(availableStorageKey) ?? window.localStorage.getItem(legacyPackedStorageKey);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    if (parsed.every((item) => typeof item === "string")) {
      return parsed.map((componentId) => {
        const spec = mvp01ComponentFlowSpecs.find((component) => component.componentId === componentId);
        const now = Date.now();
        return {
          id: componentId,
          sourceLevelId: spec?.levelId ?? "unknown",
          title: spec?.title ?? componentId,
          version: spec?.version ?? 1,
          status: "available",
          exportedModuleId: spec?.exportModuleId ?? componentId,
          implementationGraph: { levelId: spec?.levelId ?? "unknown", version: 1, nodes: [], edges: [], outputNodes: [] },
          certification: { visible: undefined, hidden: undefined },
          createdAt: now,
          updatedAt: now
        };
      });
    }
    return parsed.filter(isAvailableComponentRecord).map((record) => ({ ...record, status: "available" }));
  } catch {
    return [];
  }
}

function isAvailableComponentRecord(value: unknown): value is AvailableComponentRecord {
  if (!value || typeof value !== "object") return false;
  const record = value as Partial<AvailableComponentRecord> & { status?: string };
  return (
    typeof record.id === "string" &&
    typeof record.sourceLevelId === "string" &&
    typeof record.title === "string" &&
    (record.status === "available" || record.status === "packed") &&
    typeof record.exportedModuleId === "string" &&
    typeof record.implementationGraph === "object"
  );
}
