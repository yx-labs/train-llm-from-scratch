import type { ModuleDef } from "./types";

export class ModuleRegistry {
  private readonly modules = new Map<string, ModuleDef>();

  register(module: ModuleDef) {
    if (this.modules.has(module.id)) {
      throw new Error(`Duplicate module id: ${module.id}`);
    }
    this.modules.set(module.id, module);
  }

  get(moduleId: string) {
    const module = this.modules.get(moduleId);
    if (!module) throw new Error(`Unknown module id: ${moduleId}`);
    return module;
  }

  maybeGet(moduleId: string) {
    return this.modules.get(moduleId);
  }

  list() {
    return [...this.modules.values()];
  }
}

export function createModuleRegistry(modules: ModuleDef[]) {
  const registry = new ModuleRegistry();
  modules.forEach((module) => registry.register(module));
  return registry;
}

