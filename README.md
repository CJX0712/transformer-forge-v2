# Transformer Forge

**单文件、零依赖**地从零手写一个 decoder-only 因果 Transformer（GPT 的核心架构），在浏览器里跑通训练与自回归生成。没有 PyTorch、没有 TensorFlow、没有外部 CDN——纯 JavaScript。

> 这是「Forge」ML 实验室系列之一：每个算法都从数学公式手写实现，并用可交叉验证的**不变量**证明它真的正确，而不是“看起来能跑”。

## 它是什么

一个交互式实验室，让你亲手观察注意力机制如何学会“复制”——Transformer 在合成**复制 / 回显任务**上训练：

- 输入：`内容 + 分隔符(DELIM) + 内容`（如 `[7,2,3,6,DELIM,7,2,3,6]`）
- 目标（next-token）：在 `DELIM` 之后把原内容**回显**出来
- 训练后，喂前缀（内容 + DELIM），模型自回归地吐出 `[7,2,3,6]`

这正是自注意力“回望过去、复制信息”能力的最小可控演示。

## 从零实现（约 350 行纯 JS）

| 组件 | 说明 |
|---|---|
| Token + 正弦位置编码 | 词嵌入 + 固定正弦 PE（非学习） |
| 因果多头自注意力 | 手写前向 + 反向，因果掩码（`j > i` 强制 0） |
| 逐位置前馈网络 (FFN) | 两层线性 + ReLU |
| Pre-LayerNorm | 每层 LN 后再残差，训练更稳定 |
| 交叉熵 + Adam | 含梯度裁剪（global-norm clip）稳定训练 |

## 不变量（Headless 验证，见 `_smoke.js`）

| 不变量 | 结果 |
|---|---|
| 梯度检验：解析反向 vs 中心差分 | ✅ maxRel < 1e-4（685 个参数） |
| 注意力权重行和 = 1 | ✅ |
| 注意力严格因果（`A[i][j]=0, j>i`） | ✅ |
| 正弦 PE 精确值 + 有界 [-1,1] | ✅ |
| 训练确定性（同种子 bit 级一致） | ✅ |
| 收敛：loss 下降、回显准确率 > 随机 2× | ✅ |
| 生成：前缀 → 回显原序列（≥8/10 精确） | ✅ |
| 边界：单 token 前向无 NaN | ✅ 共 15/15 |

`_probe.js` 还会把注意力热力图（ASCII）和回显示例 dump 到 `_probe.txt`，用眼睛确认模型“真的在复制”而不是蒙对。

## 跑起来

直接用浏览器打开 `index.html` 即可（离线、零依赖）。点「训练模型」，观察：

- **loss 曲线**：交叉熵随迭代下降
- **注意力热力图**：末层 head0，每行=query、每列=key，上三角应为空白（因果）
- **生成演示**：喂前缀 → 模型回显原序列，匹配处高亮

### Headless 测试

```bash
node _smoke.js      # 15 项引擎不变量
node _uicheck.js    # DOM stub 跑通 UI 接线
node _probe.js      # 打印注意力/回显 ASCII 快照
```

## 复现训练结果

引擎在 `globalThis.TRNS` 上暴露。默认配置（d=24, heads=4, layers=2, 500 迭代, lr=0.003, seed=42）训练后回显准确率 **100%**（20/20 完全匹配，约 15s/原生 Node）。

```js
const res = TRNS.train({});              // 默认配置
console.log(res.evalAcc, res.fullMatch); // 1.0, 20
```

## License

MIT — © 晨星
