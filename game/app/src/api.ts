import type { ArenaCompareResponse, Artifact, Mission, Progress, SubmitResponse } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export function fetchMissions(): Promise<Mission[]> {
  return request<Mission[]>("/api/missions");
}

export function fetchProgress(): Promise<Progress> {
  return request<Progress>("/api/progress");
}

export function resetProgress(): Promise<Progress> {
  return request<Progress>("/api/progress/reset", { method: "POST", body: "{}" });
}

export function submitMission(missionId: string, submission: Record<string, unknown>): Promise<SubmitResponse> {
  return request<SubmitResponse>(`/api/missions/${missionId}/submit`, {
    method: "POST",
    body: JSON.stringify({ submission })
  });
}

export function fetchArtifacts(): Promise<Artifact[]> {
  return request<Artifact[]>("/api/artifacts");
}

export function compareArena(payload: {
  prompt: string;
  artifact_ids: string[];
  temperature: number;
  top_p: number;
}): Promise<ArenaCompareResponse> {
  return request<ArenaCompareResponse>("/api/arena/compare", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}
