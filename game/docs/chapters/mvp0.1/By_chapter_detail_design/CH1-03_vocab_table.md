# Chapter 1-3 Vocab Table：VocabTable

## 1. 关卡定位

- 所属章节：Chapter 1 Text → Token Pipeline
- 构建组件：`VocabTable`
- 输入来源：token string → id map
- 学习目标：token 是词表地址
- 后续用途：embedding lookup

## 2. 当前案例

VocabTable 把「token string → id map」变成可复用的图组件。

任务不是背公式，而是处理一个具体图案例：把准备好的案例数据通过 VocabTable 连接到合约探针。

案例数据输出合约：`int[V]`，示例 shape 为 `[16]`。

## 3. 玩家操作

1. 观察预制输入节点和合约探针节点。
2. 添加或修复 `VocabTable` 节点。
3. 将输入端口按语义连接到组件，再将组件输出连接到合约节点。
4. 点击「检查当前任务」确认当前案例通过。
5. 在认证变体中调整公开参数，点击「提交认证」。

## 4. 挑战设计

- 主要挑战：识别当前组件的输入/输出语义，而不是只按位置连线。
- 常见错误：漏连输入、把输出直接接到合约、忽略 dtype 或 axis 语义。
- 反馈方式：合约节点报 dtype/shape/axis 错误，Trace 面板定位第一个失败节点。

## 5. 认证变体

公开认证不要求玩家手写完整 tensor。玩家只调整少量结构参数，系统生成变体输入。

- dtype 必须保持：`int`
- axis 必须保持：`[V]`
- shape 可以随认证宽度变化，但语义不变。

## 6. 通过标准

当前任务通过：输出必须保持 dtype=int，轴为 [V]。

认证通过：同一张图在公开变体和系统变体下仍满足组件合约，组件变为可用并进入下一关。
