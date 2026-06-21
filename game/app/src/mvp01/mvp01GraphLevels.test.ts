import { describe, expect, it } from "vitest";
import { graphText } from "../gameplayGraph/i18n";
import { createGameplayRegistry } from "../gameplayGraph/modules";
import { runTests } from "../gameplayGraph/runtime/testRunner";
import { mvp01ComponentFlowSpecs, mvp01GraphLevels } from "./mvp01GraphLevels";

const registry = createGameplayRegistry();

describe("MVP0.1 graph component arc", () => {
  it("passes visible and hidden certification tests with every target graph", () => {
    mvp01GraphLevels.forEach((level) => {
      expect(level.targetGraph, `${level.id} should define a target graph`).toBeDefined();
      const graph = level.targetGraph ?? level.initialGraph;
      const visible = runTests(graph, registry, level.visibleTests);
      const hidden = runTests(graph, registry, level.hiddenTests);

      expect(visible.status, `${level.id} visible`).toBe("pass");
      expect(hidden.status, `${level.id} hidden`).toBe("pass");
    });
  });

  it("does not ship the initial graphs as already certified", () => {
    mvp01GraphLevels.forEach((level) => {
      const visible = runTests(level.initialGraph, registry, level.visibleTests);

      expect(visible.status, `${level.id} initial graph`).not.toBe("pass");
    });
  });

  it("uses available component modules as later blueprint dependencies", () => {
    const specs = Object.fromEntries(mvp01ComponentFlowSpecs.map((component) => [component.levelId, component]));
    const tutorial = mvp01GraphLevels.find((level) => level.id === "mvp01_0_wire_probe");
    const scalar = mvp01GraphLevels.find((level) => level.id === "mvp01_1_scalar_cell");
    const vector = mvp01GraphLevels.find((level) => level.id === "mvp01_2_vector_rail");
    const linear = mvp01GraphLevels.find((level) => level.id === "mvp01_6_linear");

    expect(mvp01GraphLevels[0].id).toBe("mvp01_0_wire_probe");
    expect(tutorial?.modulePalette).toEqual([]);
    expect(specs.mvp01_0_wire_probe).toBeUndefined();
    expect(scalar?.modulePalette).toContain("Float32Literal");
    expect(scalar?.modulePalette).not.toContain("ScalarCell");
    expect(scalar?.targetGraph?.nodes.find((node) => node.id === "scalar_source")?.moduleId).toBe("Float32Literal");

    expect(mvp01GraphLevels).toHaveLength(74);
    expect(mvp01GraphLevels.map((level) => level.title)).toContain("Chapter 9-6 Tiny Chat Loop");
    expect(specs.mvp01_ch0_02_scalar_add.requires).toContain("component.scalar_cell.v1");
    expect(specs.mvp01_2_vector_rail.requires).toContain("component.scalar_add.v1");
    expect(vector?.modulePalette).toContain("component.scalar_cell.v1");
    expect(vector?.modulePalette).not.toContain("ScalarCell");

    expect(specs.mvp01_6_linear.requires).toContain("component.bias_add.v1");
    expect(linear?.modulePalette).toContain("component.matmul_gate.v1");
    expect(linear?.modulePalette).not.toContain("MatMulGate");
    expect(linear?.targetGraph?.nodes.find((node) => node.id === "matmul")?.moduleId).toBe("component.matmul_gate.v1");
  });

  it("defines public certification variants for every buildable MVP0.1 component", () => {
    const buildableLevels = mvp01GraphLevels.filter((level) => mvp01ComponentFlowSpecs.some((component) => component.levelId === level.id));

    buildableLevels.forEach((level) => {
      expect(level.certification, `${level.id} certification`).toBeDefined();
      expect(level.certification?.controls.length, `${level.id} controls`).toBeGreaterThan(0);
      const values = Object.fromEntries(level.certification!.controls.map((control) => [control.id, control.defaultValue]));
      const publicTests = level.certification!.makePublicTests(level.targetGraph ?? level.initialGraph, values);

      expect(publicTests.length, `${level.id} public tests`).toBeGreaterThan(0);
      publicTests.forEach((item) => {
        const result = runTests(item.graph ?? level.targetGraph ?? level.initialGraph, registry, [item.testCase]);
        expect(result.status, `${level.id} ${item.testCase.id}`).toBe("pass");
      });
    });
  });

  it("certifies target graphs with public variants plus system variants", () => {
    mvp01GraphLevels.forEach((level) => {
      if (!level.certification) return;
      const graph = level.targetGraph ?? level.initialGraph;
      const values = Object.fromEntries(level.certification.controls.map((control) => [control.id, control.defaultValue]));
      const publicTests = level.certification.makePublicTests(graph, values);
      const certificationRuns = [...publicTests, ...level.hiddenTests.map((testCase) => ({ testCase, graph: undefined }))];

      certificationRuns.forEach((item) => {
        const result = runTests(item.graph ?? graph, registry, [item.testCase]);
        expect(result.status, `${level.id} ${item.testCase.id}`).toBe("pass");
      });
    });
  });

  it("uses generated certification controls instead of handwritten large tensor inputs", () => {
    const advancedLevels = ["mvp01_3_matrix_struct", "mvp01_4_tensor_box", "mvp01_5_matmul_gate", "mvp01_6_linear"];

    advancedLevels.forEach((levelId) => {
      const level = mvp01GraphLevels.find((item) => item.id === levelId);
      expect(level?.certification).toBeDefined();
      const controlIds = level!.certification!.controls.map((control) => control.id);

      expect(controlIds).toContain("seed");
      expect(level!.certification!.controls.every((control) => control.kind === "integer" || control.kind === "select")).toBe(true);
    });
  });

  it("teaches the prebuilt probe as test equipment before ScalarCell construction", () => {
    const tutorial = mvp01GraphLevels.find((level) => level.id === "mvp01_0_wire_probe");
    expect(tutorial).toBeDefined();
    if (!tutorial) throw new Error("missing MVP0.1-0 tutorial");
    expect(tutorial.onboarding).toBeDefined();
    if (!tutorial.onboarding) throw new Error("missing MVP0.1-0 onboarding");

    const copy = [tutorial.goal, tutorial.caseStudy?.narrative, tutorial.debrief?.learned].join("\n");
    expect(copy).toMatch(/prebuilt probe|prebuilt test equipment/i);
    expect(copy).toMatch(/not.*reusable component/i);
    expect(copy).not.toMatch(/rank-0|finite float32 literal/i);
    expect(tutorial.onboarding.targetRecipe).toEqual(["packet_input.out -> port_gate.x", "port_gate.out -> reference.x"]);
    expect(tutorial.targetGraph?.edges.map((edge) => `${edge.from.nodeId}.${edge.from.portId} -> ${edge.to.nodeId}.${edge.to.portId}`)).toEqual([
      "packet_input.out -> port_gate.x",
      "port_gate.out -> reference.x"
    ]);
    expect(tutorial.targetGraph?.nodes.find((node) => node.id === "packet_input")?.moduleId).toBe("InputTensor");
  });

  it("keeps MVP0.1 level copy off the removed manual availability step", () => {
    const copy = mvp01GraphLevels
      .map((level) =>
        [
          level.goal,
          level.onboarding?.story,
          level.onboarding?.startingProblem,
          level.onboarding?.firstAction,
          level.caseStudy?.narrative,
          level.caseStudy?.visibleInputFocus,
          level.debrief.completeTitle,
          level.debrief.fixedProblem,
          level.debrief.learned,
          level.debrief.nextUse
        ].join("\n")
      )
      .join("\n");

    expect(copy).not.toMatch(/\bpack\b|\bpacked\b|\bpackable\b|\bpackaging\b|\bpackage\b/i);
  });

  it("keeps every MVP0.1 level case-first with readable scenario text and concrete tensor data", () => {
    mvp01GraphLevels.forEach((level) => {
      const panelTypes = level.caseStudy?.dataPanels.map((panel) => panel.type) ?? [];

      expect(level.caseStudy?.title, `${level.id} case title`).toBeTruthy();
      expect(level.caseStudy?.narrative, `${level.id} case narrative`).toBeTruthy();
      expect(panelTypes, `${level.id} data panels`).toContain("text_batch");
      expect(panelTypes, `${level.id} data panels`).toContain("tensor_preview");
      expect(level.visibleTests[0].inputs.case?.dtype, `${level.id} readable case input`).toBe("raw_text");
    });
  });

  it("keeps the first Scalar case scoped to the scalar concept only", () => {
    const scalar = mvp01GraphLevels.find((level) => level.id === "mvp01_1_scalar_cell");
    const caseLines = scalar?.visibleTests[0].inputs.case?.data;
    const caseText = Array.isArray(caseLines) ? caseLines.join("\n") : String(caseLines ?? "");

    expect(caseText).toContain("Scalar");
    expect(caseText).not.toMatch(/residual|prompt|later|model/i);
    expect(scalar?.caseStudy?.title).toBe("Create a valid Scalar");
    expect(graphText("zh", scalar?.caseStudy?.title ?? "")).toContain("标量");
    expect(graphText("zh", Array.isArray(caseLines) ? String(caseLines[0]) : "")).toContain("标量");
  });

  it("keeps MVP0.1 case copy bilingual through the shared graphText table", () => {
    mvp01GraphLevels.forEach((level) => {
      expect(level.caseStudy?.title, `${level.id} case title`).toBeTruthy();
      expect(graphText("en", level.caseStudy!.title), `${level.id} english title`).toBe(level.caseStudy!.title);
      expect(graphText("zh", level.caseStudy!.title), `${level.id} chinese title`).not.toBe(level.caseStudy!.title);

      const caseLines = level.visibleTests[0].inputs.case?.data;
      expect(Array.isArray(caseLines), `${level.id} case lines`).toBe(true);
      (caseLines as string[]).forEach((line) => {
        expect(graphText("zh", line), `${level.id} zh line ${line}`).not.toBe(line);
      });
    });
  });

  it("rejects a ScalarCell value edit when the player enters a non-float32 value", () => {
    const scalar = mvp01GraphLevels.find((level) => level.id === "mvp01_1_scalar_cell");
    expect(scalar?.targetGraph).toBeDefined();
    const graph = {
      ...(scalar?.targetGraph ?? scalar!.initialGraph),
      nodes: (scalar?.targetGraph ?? scalar!.initialGraph).nodes.map((node) =>
        node.id === "scalar_source" ? { ...node, params: { ...node.params, value: "not-a-number" } } : node
      )
    };

    const visible = runTests(graph, registry, scalar!.visibleTests);

    expect(visible.status).toBe("blocked");
    expect(visible.results.some((result) => result.firstBadNodeId === "scalar_source" && result.diagnostic?.errorType === "nan_inf")).toBe(true);
  });

  it("rejects a non-float32 public certification value instead of falling back to the default", () => {
    const scalar = mvp01GraphLevels.find((level) => level.id === "mvp01_1_scalar_cell");
    expect(scalar?.targetGraph).toBeDefined();
    expect(scalar?.certification).toBeDefined();
    const graph = scalar!.targetGraph ?? scalar!.initialGraph;
    const [publicVariant] = scalar!.certification!.makePublicTests(graph, { value: "not-a-number" });

    const result = runTests(publicVariant.graph ?? graph, registry, [publicVariant.testCase]);

    expect(result.status).toBe("blocked");
    expect(result.results.some((item) => item.firstBadNodeId === "scalar_source" && item.diagnostic?.errorType === "nan_inf")).toBe(true);
  });

  it("accepts any finite ScalarCell value instead of matching only the starting example", () => {
    const scalar = mvp01GraphLevels.find((level) => level.id === "mvp01_1_scalar_cell");
    expect(scalar?.targetGraph).toBeDefined();
    expect(scalar?.visibleTests[0].assertions.some((assertion) => assertion.type === "allclose")).toBe(false);
    const graph = {
      ...(scalar?.targetGraph ?? scalar!.initialGraph),
      nodes: (scalar?.targetGraph ?? scalar!.initialGraph).nodes.map((node) =>
        node.id === "scalar_source" ? { ...node, params: { ...node.params, value: "42" } } : node
      )
    };

    const visible = runTests(graph, registry, scalar!.visibleTests);

    expect(visible.status).toBe("pass");
  });
});
