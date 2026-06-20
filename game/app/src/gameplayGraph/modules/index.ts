import { createModuleRegistry } from "../moduleRegistry";
import { tensorModules } from "./tensorModules";
import { tokenizerModules } from "./tokenizerModules";

export const gameplayModules = [...tensorModules, ...tokenizerModules];

export function createGameplayRegistry() {
  return createModuleRegistry(gameplayModules);
}

