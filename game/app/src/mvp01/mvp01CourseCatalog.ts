import rawCourseLevels from "./mvp01CourseCatalog.json";
import type { AxisName, DType } from "../gameplayGraph/types";

export type Mvp01CourseLevelOutput = {
  dtype: DType;
  dims: number[];
  axes: AxisName[];
};

export type Mvp01CourseLevelDef = {
  code: string;
  id: string;
  chapter: number;
  chapterTitle: string;
  chapterTitleZh: string;
  chapterGoalZh: string;
  index: number;
  slug: string;
  title: string;
  titleZh: string;
  component: string;
  componentZh: string;
  componentId: string;
  moduleId: string;
  buildable: boolean;
  implementedManually: boolean;
  from: string;
  fromZh: string;
  learn: string;
  learnZh: string;
  futureUse: string;
  futureUseZh: string;
  output: Mvp01CourseLevelOutput;
  shelf: string;
  goal: string;
  goalZh: string;
  task: string;
  taskZh: string;
  concept: string;
  conceptZh: string;
  check: string;
  checkZh: string;
  requires: string[];
  unlocks: string[];
};

export const mvp01CourseLevels = rawCourseLevels as Mvp01CourseLevelDef[];

export const mvp01ManualCourseCodes = new Set(mvp01CourseLevels.filter((level) => level.implementedManually).map((level) => level.code));

export const mvp01CourseZhText: Record<string, string> = Object.fromEntries(
  mvp01CourseLevels.flatMap((level) => [
    [`Chapter ${level.code} ${level.title}`, `Chapter ${level.code} ${level.titleZh.replace(`${level.code} `, "")}`],
    [level.chapterTitle, level.chapterTitleZh],
    [level.goal, level.goalZh],
    [level.task, level.taskZh],
    [level.concept, level.conceptZh],
    [level.check, level.checkZh],
    [`Concept: ${level.concept}`, `概念：${level.conceptZh}`],
    [`Task: ${level.task}`, `任务：${level.taskZh}`],
    [`Check: ${level.check}`, `检查：${level.checkZh}`],
    [`Build ${level.component}`, `构造 ${level.componentZh}`],
    [`Validate ${level.component}`, `验证 ${level.componentZh}`],
    [`${level.component} contract`, `${level.componentZh} 合约`]
  ])
);

const threeInputSlugs = new Set(["linear_module", "attention_head", "multi_head_attention", "transformer_block", "tiny_chat_loop"]);
const twoInputSlugs = new Set([
  "scalar_add",
  "dot_product",
  "matrix_multiply",
  "embedding_lookup",
  "hidden_init",
  "bias_add",
  "qk_score",
  "mask_apply",
  "weighted_sum",
  "head_concat",
  "output_projection",
  "residual_add",
  "attention_sublayer",
  "mlp_sublayer",
  "block_stack",
  "cross_entropy_cell",
  "forward_runner",
  "backward_trace",
  "optimizer",
  "checkpoint",
  "append_token"
]);

export function mvp01CourseInputCount(level: Mvp01CourseLevelDef) {
  if (threeInputSlugs.has(level.slug)) return 3;
  if (twoInputSlugs.has(level.slug)) return 2;
  return 1;
}
