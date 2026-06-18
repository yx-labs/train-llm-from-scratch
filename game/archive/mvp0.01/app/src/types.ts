export type MissionStatus = "locked" | "unlocked" | "completed";

export type Mission = {
  id: string;
  order: number;
  stage: string;
  chapter: string;
  title: string;
  type: string;
  summary: string;
  learning_objectives: string[];
  prerequisites: string[];
  unlocks: string[];
  unlock_artifacts?: string[];
  inputs: Record<string, unknown>;
  default_submission: Record<string, unknown>;
  success: Record<string, unknown>;
  failure_rules: string[];
  hints: string[];
  concept?: string;
  status: MissionStatus;
};

export type Progress = {
  player_id: string;
  completed_missions: string[];
  mission_progress: Record<string, { status: string; attempts: number; best_score: number | null }>;
  unlocked_modules: string[];
  unlocked_artifacts: string[];
  notebook: Array<{ concept: string; status: string }>;
};

export type IncidentReport = {
  id?: string;
  title: string;
  severity?: string;
  symptoms?: string[];
  suspected_modules?: string[];
  hints?: string[];
};

export type MetricPoint = {
  step: number;
  train_loss: number;
  val_loss: number;
  gpu_memory_mb: number;
};

export type SubmitResponse = {
  mission_id: string;
  passed: boolean;
  unlocked_modules: string[];
  unlocked_artifacts: string[];
  next_mission_id: string | null;
  progress: Progress;
  incident_report?: IncidentReport | null;
  metrics?: MetricPoint[] | null;
  events?: Array<{ step: number; type: string; severity: string }>;
  preview?: string | null;
};

export type Artifact = {
  id: string;
  stage: string;
  kind: string;
  display_name: string;
  description: string;
  capabilities: string[];
  unlocked_by: string;
  unlocked: boolean;
  source: string;
};

export type ArenaCompareResponse = {
  prompt: string;
  generation: {
    temperature: number;
    top_p: number;
  };
  results: Array<{
    artifact_id: string;
    display_name: string;
    stage?: string;
    output: string;
    notes: string;
    source: string;
    unlocked: boolean;
  }>;
};
