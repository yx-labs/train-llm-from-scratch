import { createModuleRegistry } from "../moduleRegistry";
import { tensorModules } from "./tensorModules";
import { tokenizerModules } from "./tokenizerModules";
import { mvp01ComponentModules } from "./mvp01ComponentModules";

export const gameplayModules = [...tensorModules, ...tokenizerModules, ...mvp01ComponentModules];

export function createGameplayRegistry() {
  return createModuleRegistry(gameplayModules);
}
