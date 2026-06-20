import { ModuleRegistry } from "../moduleRegistry";
import type { GraphExecutionResult, GraphSpec, TestCase, TestResult, TestVisibility } from "../types";
import { executeGraph } from "./graphExecutor";
import { evaluateAssertion } from "./assertions";

export type RunTestsResult = {
  status: "pass" | "fail" | "blocked";
  results: TestResult[];
  cases: TestCaseRunResult[];
};

export type TestCaseRunResult = {
  id: string;
  title: string;
  visibility: TestVisibility;
  status: "pass" | "fail" | "blocked";
  results: TestResult[];
  execution: GraphExecutionResult;
};

export function runTestCaseDetailed(graph: GraphSpec, registry: ModuleRegistry, testCase: TestCase): TestCaseRunResult {
  const execution = executeGraph(graph, registry, testCase.inputs);
  const results = testCase.assertions.map((assertion, index) =>
    evaluateAssertion(assertion, execution, testCase.visibility, `${testCase.id}:${index}`)
  );
  return {
    id: testCase.id,
    title: testCase.title,
    visibility: testCase.visibility,
    status: statusFromResults(results),
    results,
    execution
  };
}

export function runTestCase(graph: GraphSpec, registry: ModuleRegistry, testCase: TestCase): TestResult[] {
  return runTestCaseDetailed(graph, registry, testCase).results;
}

export function runTests(graph: GraphSpec, registry: ModuleRegistry, tests: TestCase[]): RunTestsResult {
  const cases = tests.map((testCase) => runTestCaseDetailed(graph, registry, testCase));
  const results = cases.flatMap((testCase) => testCase.results);
  return { status: statusFromResults(results), results, cases };
}

function statusFromResults(results: TestResult[]) {
  return results.some((result) => result.status === "fail")
    ? "fail"
    : results.some((result) => result.status === "blocked")
      ? "blocked"
      : "pass";
}
